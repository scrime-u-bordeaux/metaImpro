import os
import json
import hashlib
from typing import List, Dict, Any
from collections import defaultdict
from midi_processor import MidiSymbolProcessor
from factor_oracle import OracleBuilder
from accompaniement import make_vlmc_for_chord, transpose_chord_data, get_pitches_by_chord
from markov import build_vlmc_table, symbol_to_key

# ---------------- JSON <-> complex structure helpers ----------------

def _serialize_node(obj: Any) -> Any:
    """Convert complex python object into JSON-serializable structure with tags."""
    # primitives
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj

    # tuple
    if isinstance(obj, tuple):
        return {'__tuple__': True, 'items': [_serialize_node(i) for i in obj]}

    # list
    if isinstance(obj, list):
        return [_serialize_node(i) for i in obj]

    # defaultdict
    if isinstance(obj, defaultdict):
        factory_tag = 'callable'
        f = obj.default_factory
        if f is int:
            factory_tag = 'int'
        elif f is list:
            factory_tag = 'list'
        elif f is None:
            factory_tag = 'none'
        else:
            # try detect common lambda: defaultdict(int)
            try:
                sample = f()
                if isinstance(sample, defaultdict) and sample.default_factory is int:
                    factory_tag = 'defaultdict_int'
            except Exception:
                pass

        items = []
        for k, v in obj.items():
            items.append([_serialize_node(k), _serialize_node(v)])
        return {'__defaultdict__': True, 'default_factory': factory_tag, 'items': items}

    # normal dict (preserve mapping for string keys, otherwise list-of-pairs)
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj.keys()):
            return {k: _serialize_node(v) for k, v in obj.items()}
        items = []
        for k, v in obj.items():
            items.append([_serialize_node(k), _serialize_node(v)])
        return {'__mapping__': True, 'items': items}

    # fallback for other objects: store repr
    return {'__repr__': True, 'value': repr(obj)}


def _deserialize_node(node: Any) -> Any:
    """Reconstruct Python object from the tagged JSON-friendly structure."""
    if node is None or isinstance(node, (str, bool, int, float)):
        return node

    if isinstance(node, list):
        return [_deserialize_node(i) for i in node]

    if isinstance(node, dict) and node.get('__tuple__'):
        return tuple(_deserialize_node(i) for i in node['items'])

    if isinstance(node, dict) and node.get('__defaultdict__'):
        tag = node.get('default_factory', 'callable')
        items = node.get('items', [])
        if tag == 'int':
            dd = defaultdict(int)
        elif tag == 'list':
            dd = defaultdict(list)
        elif tag == 'none':
            dd = defaultdict(lambda: None)
        elif tag == 'defaultdict_int':
            dd = defaultdict(lambda: defaultdict(int))
        else:
            dd = defaultdict(dict)
        for k_node, v_node in items:
            k = _deserialize_node(k_node)
            v = _deserialize_node(v_node)
            dd[k] = v
        return dd

    if isinstance(node, dict) and node.get('__mapping__'):
        out = {}
        for k_node, v_node in node.get('items', []):
            k = _deserialize_node(k_node)
            v = _deserialize_node(v_node)
            out[k] = v
        return out

    # regular dict with string keys (no special tags)
    if isinstance(node, dict) and not any(k.startswith('__') for k in node.keys()):
        return {k: _deserialize_node(v) for k, v in node.items()}

    if isinstance(node, dict) and node.get('__repr__'):
        # Can't reconstruct arbitrary object — return repr string
        return node.get('value')

    # fallback: attempt to rebuild dict-like structure
    return {k: _deserialize_node(v) for k, v in node.items()}


def convert_to_json(data: Any) -> Any:
    """
    Convert complex structure to JSON-serializable Python object.
    Use json.dump(convert_to_json(obj), ...) to write to disk.
    """
    return _serialize_node(data)


def convert_from_json_data(loaded: Any) -> Any:
    """
    Convert Python object produced by json.load(...) of a convert_to_json output
    back to the original complex structure.
    """
    return _deserialize_node(loaded)


# ---------------- simple cache key helper ----------------

def get_cache_key(*args, **kwargs) -> str:
    """Generate a stable cache key from args/kwargs."""
    key_data = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_data.encode()).hexdigest()


# ----------------- corpus & symbols loading (with caching) -----------------

def load_corpus(input_path: str) -> List[dict]:
    """
    Load corpus from file or MIDI. Internal helper used by cached version.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.json':
        with open(input_path, 'r', encoding='utf-8') as f:
            symbols = json.load(f)
        if not isinstance(symbols, list):
            raise ValueError("JSON must contain a list of symbols")
    elif ext in (".mid", ".midi"):
        symbols = MidiSymbolProcessor().process_midi_file(input_path)
        if not symbols:
            raise ValueError(f"No symbols generated for {input_path}")
    elif ext == ".pt":
        # implement if required
        symbols = []
    else:
        raise ValueError(f"Unsupported corpus extension: {ext}")
    return symbols


def load_corpus_cached(input_path: str) -> List[dict]:
    """
    Load corpus with simple caching keyed on file modification time.
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    try:
        file_mtime = os.path.getmtime(input_path)
        cache_key = get_cache_key(input_path, file_mtime)
    except OSError:
        cache_key = get_cache_key(input_path)

    cache_file = os.path.join(cache_dir, f"corpus_{cache_key}.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    symbols = load_corpus(input_path)

    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(symbols, f, indent=2)
    except IOError:
        pass

    return symbols


def load_symbols_cached(input_path: str, mode: str, markov_order: int, similarity_level: int,
                        xml_folder: str, progression: List[str], accomp_info) -> Dict[str, Any]:
    """
    Load symbols with caching. For accompaniment mode:
      - chord_map is cached (convert_to_json/convert_from_json_data used)
      - vlmcs are computed fresh or cached separately (vlmcs_{params}.json) — adjust as needed.
    For non-accompaniment modes the full result is cached.
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    if mode == "accompagnement":
        # chord cache key (depends on xml_folder + progression + accomp_info)
        chord_cache_key = get_cache_key(xml_folder, tuple(progression), accomp_info)
        chord_cache_file = os.path.join(cache_dir, f"chords_{chord_cache_key}.json")

        chord_map = None
        if os.path.exists(chord_cache_file):
            try:
                with open(chord_cache_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    print("loading chord_map")
                chord_map = convert_from_json_data(loaded)
            except (json.JSONDecodeError, IOError):
                chord_map = None

        if chord_map is None:
            # build chord_map according to accomp_info
            if accomp_info == "normal":
                chord_map = get_pitches_by_chord(xml_folder, progression)
            elif accomp_info == "by chord type":
                chord_map = get_pitches_by_chord(xml_folder, progression, group_by_type=True, max_notes_per_chord=600)
                chord_map = transpose_chord_data(chord_map, progression)
            elif accomp_info == "by interval":
                chord_map = {}
            else:
                chord_map = {}

            # save chord_map serialized
            try:
                with open(chord_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_to_json(chord_map), f, indent=2)
            except IOError:
                pass

        result = {
            'symbols': [],
            'chord_map': chord_map,
            'progression': progression
        }

        # VLMC cache key (includes markov params since VLMC depends on them)
        vlmc_cache_key = get_cache_key(xml_folder, tuple(progression), accomp_info, markov_order, similarity_level)
        vlmc_cache_file = os.path.join(cache_dir, f"vlmcs_{vlmc_cache_key}.json")

        vlmcs = None
        if os.path.exists(vlmc_cache_file):
            try:
                with open(vlmc_cache_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    print("loading vlmcs")
                vlmcs = convert_from_json_data(loaded)
            except (json.JSONDecodeError, IOError):
                vlmcs = None

        if vlmcs is None:
            # compute vlmcs fresh
            if accomp_info in ("normal", "by chord type"):
                vlmcs = make_vlmc_for_chord(
                    chord_map,
                    max_order=markov_order,
                    similarity_level=similarity_level,
                )
                # try to save vlmcs serialized (can be large)
                try:
                    with open(vlmc_cache_file, 'w', encoding='utf-8') as f:
                        json.dump(convert_to_json(vlmcs), f, indent=2)
                except (IOError, TypeError):
                    # if saving fails, continue without persistent caching
                    pass
            else:
                vlmcs = {}

        result['vlmcs'] = vlmcs
        return result

    else:
        # non-accompagnement: full caching of the result
        cache_key = get_cache_key(input_path, mode, markov_order, similarity_level, xml_folder, tuple(progression), accomp_info)
        cache_file = os.path.join(cache_dir, f"symbols_{cache_key}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                return convert_from_json_data(cached_data)
            except (json.JSONDecodeError, IOError):
                pass

        # generate fresh result
        result = load_symbols(input_path, mode, markov_order, similarity_level, xml_folder, progression, accomp_info)

        # save serialized version
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(convert_to_json(result), f, indent=2)
        except (IOError, TypeError):
            pass

        return result


# ----------------- original load_symbols function (unchanged logic) -----------------

def load_symbols(input_path: str, mode: str, markov_order: int, similarity_level: int, xml_folder: str,
                 progression: List[str], accomp_info) -> Dict[str, Any]:
    """
    Original load_symbols function - used internally by cached version.
    """
    if mode == "Autoencoder":
        return {
            'symbols': [],
            'model_path': input_path,
            'config_path': "piano_genie/cfg.json"
        }

    symbols = load_corpus(input_path)
    result: Dict[str, Any] = {'symbols': symbols}

    if mode == 'oracle':
        oracle_result = OracleBuilder.build_oracle(symbols)
        trans_list = [oracle_result[i] for i in range(0, len(oracle_result), 2)]
        supp_list = [oracle_result[i] for i in range(1, len(oracle_result), 2)]

        level_map = {3: 0, 2: 1, 1: 2}
        idx = level_map[similarity_level]
        result['trans_oracle'] = trans_list[idx]
        result['supply'] = supp_list[idx]

    if mode == 'markov':
        vlmc_table = build_vlmc_table(symbols, max_order=markov_order, similarity_level=similarity_level)
        all_keys = list({symbol_to_key(s) for s in symbols})
        result['vlmc_table'] = vlmc_table
        result['notes'] = all_keys

    if mode in ('markov', 'random'):
        result['unique_pitches'] = []
        seen = set()
        for s in symbols:
            if s.get('type') == 'note':
                p = s.get('pitch')
                if p not in seen:
                    result['unique_pitches'].append(p)
                    seen.add(p)
            elif s.get('type') == 'chord':
                result['unique_pitches'].append(tuple(s.get('pitch', [])))

    if mode == "accompagnement":
        if accomp_info == "normal":
            chord_map = get_pitches_by_chord(xml_folder, progression)
            result['chord_map'] = chord_map
            result['vlmcs'] = make_vlmc_for_chord(
                chord_map,
                max_order=markov_order,
                similarity_level=similarity_level,
            )

        elif accomp_info == "by chord type":
            chord_map = get_pitches_by_chord(xml_folder, progression, group_by_type=True, max_notes_per_chord=600)
            chord_map = transpose_chord_data(chord_map, progression)
            result['chord_map'] = chord_map
            result['vlmcs'] = make_vlmc_for_chord(
                chord_map,
                max_order=markov_order,
                similarity_level=similarity_level,
            )

        elif accomp_info == "by interval":
            # placeholder: your original commented-out code can be placed here if needed
            pass

    result['progression'] = progression
    return result

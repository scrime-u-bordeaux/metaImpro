import os
import json
import hashlib
from typing import List, Dict, Any
from collections import defaultdict
from midi_processor import MidiSymbolProcessor
from factor_oracle import OracleBuilder
from accompaniement import make_vlmc_for_chord, transpose_chord_data, get_pitches_by_chord
from markov import build_vlmc_table, symbol_to_key

def convert_to_json(data):
    """Convert complex nested structure to JSON-serializable format."""
    result = {}
    
    for key, value in data.items():
        # Convert tuple keys to string representation
        key_str = str(key)
        
        # Convert defaultdict values to regular dict
        if isinstance(value, defaultdict):
            # Convert the defaultdict to a regular dict and handle tuple keys in nested structure
            value_dict = {}
            for inner_key, inner_value in value.items():
                inner_key_str = str(inner_key) if isinstance(inner_key, tuple) else inner_key
                value_dict[inner_key_str] = inner_value
            
            # Store both the dict and the default_factory type info
            result[key_str] = {
                'type': 'defaultdict',
                'default_factory': value.default_factory.__name__ if value.default_factory else None,
                'data': value_dict
            }
        else:
            result[key_str] = value
    
    return result

def convert_from_json_data(loaded):
    """Convert JSON-serializable format back to original complex structure."""
    result = {}
    for key_str, value in loaded.items():
        # Convert string key back to tuple using eval (be careful with eval!)
        # For production code, consider using ast.literal_eval for safety
        try:
            key = eval(key_str)
        except:
            key = key_str  # If eval fails, use string as-is
        
        if isinstance(value, dict) and value.get('type') == 'defaultdict':
            # Reconstruct defaultdict
            default_factory_name = value.get('default_factory')
            if default_factory_name == 'int':
                default_factory = int
            elif default_factory_name == 'list':
                default_factory = list
            else:
                default_factory = None
            
            # Create defaultdict and populate with data
            defaultdict_obj = defaultdict(default_factory)
            for inner_key_str, inner_value in value['data'].items():
                # Convert string key back to tuple
                try:
                    inner_key = eval(inner_key_str)
                except:
                    inner_key = inner_key_str
                defaultdict_obj[inner_key] = inner_value
            
            result[key] = defaultdict_obj
        else:
            result[key] = value
    
    return result

def get_cache_key(*args, **kwargs):
    """Generate a unique cache key based on function arguments."""
    # Convert all arguments to strings and create a hash
    key_data = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_data.encode()).hexdigest()

def load_corpus_cached(input_path: str) -> List[dict]:
    """
    Charge et renvoie la liste des symboles Midi traités avec mise en cache.
    """
    # Create cache directory if it doesn't exist
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on input path and file modification time
    try:
        file_mtime = os.path.getmtime(input_path)
        cache_key = get_cache_key(input_path, file_mtime)
    except OSError:
        # If file doesn't exist or can't get mtime, use just the path
        cache_key = get_cache_key(input_path)
    
    cache_file = os.path.join(cache_dir, f"corpus_{cache_key}.json")
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If cache file is corrupted, we'll regenerate
            pass
    
    # Generate the data if not in cache
    symbols = load_corpus(input_path)
    
    # Save to cache
    try:
        with open(cache_file, 'w') as f:
            json.dump(symbols, f, indent=2)
    except IOError:
        # If we can't write to cache, just continue without caching
        pass
    
    return symbols

def load_symbols_cached(input_path: str, mode: str, markov_order: int, similarity_level: int, 
                       xml_folder: str, progression: List[str], accomp_info) -> Dict[str, Any]:
    """
    Load symbols with caching for complex data structures.
    For accompagnement mode, only cache chord_map and build VLMCs fresh each time.
    """
    # Create cache directory if it doesn't exist
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # For accompagnement mode, we only want to cache the chord_map, not the VLMCs
    if mode == "accompagnement":
        # Generate cache key for just the chord-related data
        cache_key = get_cache_key(xml_folder, tuple(progression), accomp_info)
        cache_file = os.path.join(cache_dir, f"chords_{cache_key}.json")
        
        # Try to load chord_map from cache first
        chord_map = None
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    chord_map = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # If not in cache, generate the chord_map
        if chord_map is None:
            if accomp_info == "normal":
                chord_map = get_pitches_by_chord(xml_folder, progression)
            elif accomp_info == "by chord type":
                chord_map = get_pitches_by_chord(xml_folder, progression, group_by_type=True, max_notes_per_chord=600)
                chord_map = transpose_chord_data(chord_map, progression)
            elif accomp_info == "by interval":
                # Handle interval case if needed
                chord_map = {}
            
            # Save chord_map to cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(chord_map, f, indent=2)
            except IOError:
                pass
        
        # Build the result with fresh VLMCs
        result = {
            'symbols': [],
            'chord_map': chord_map,
            'progression': progression
        }
        
        # Build VLMCs fresh (don't cache these complex structures)
        if accomp_info in ("normal", "by chord type"):
            result['vlmcs'] = make_vlmc_for_chord(
                chord_map,
                max_order=markov_order,
                similarity_level=similarity_level,
            )
        
        return result
    
    else:
        # For non-accompagnement modes, use full caching
        cache_key = get_cache_key(input_path, mode, markov_order, similarity_level, 
                                 xml_folder, tuple(progression), accomp_info)
        cache_file = os.path.join(cache_dir, f"symbols_{cache_key}.json")
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                # Convert back from JSON-serializable format
                return convert_from_json_data(cached_data)
            except (json.JSONDecodeError, IOError):
                # If cache file is corrupted, we'll regenerate
                pass
        
        # Generate the data if not in cache
        result = load_symbols(input_path, mode, markov_order, similarity_level, 
                             xml_folder, progression, accomp_info)
        
        # Convert to JSON-serializable format and save to cache
        try:
            json_data = convert_to_json(result)
            with open(cache_file, 'w') as f:
                json.dump(json_data, f, indent=2)
        except (IOError, TypeError):
            # If we can't write to cache or convert to JSON, just continue without caching
            pass
        
        return result

# Keep your original functions as they are, just rename them for internal use
def load_corpus(input_path: str) -> List[dict]:
    """
    Original load_corpus function - used internally by cached version.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.json':
        with open(input_path, 'r') as f:
            symbols = json.load(f)
        if not isinstance(symbols, list):
            raise ValueError("JSON must contain a list of symbols")
    elif ext == ".mid" or ext == ".midi":
        symbols = MidiSymbolProcessor().process_midi_file(input_path)
        if not symbols:
            raise ValueError(f"No symbols generated for {input_path}")
    elif ext == ".pt":
        pass
    else:
        raise ValueError(f"Extension de corpus non supportée : {ext}")
    return symbols

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
            if s['type'] == 'note':
                p = s['pitch']
                if p not in seen:
                    result['unique_pitches'].append(p)
                    seen.add(p)
            elif s['type'] == 'chord':
                result['unique_pitches'].append(tuple(s['pitch']))
                
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
            pass
            """chord_map = get_pitches_by_chord(
                xml_folder,
                progression,
                group_by_type=True,
                max_notes_per_chord=600
            )
            chord_map = transpose_chord_data(chord_map, progression)
            interval_dict = build_interval_dict(chord_map)
            interval_table = build_interval_table(interval_dict, max_order=markov_order)
            
            result['chord_map'] = chord_map
            result['interval_dict'] = interval_dict
            result['interval_table'] = interval_table
            result['vlmcs'] = make_vlmc_for_chord(
                chord_map,
                max_order=markov_order,
                similarity_level=similarity_level,
                use_intervals=True
            )"""
        
    result['progression'] = progression
    return result
"""
Utility script to save argparse arguments to a config.json file.

Usage:
    from save_config import save_args_as_config
    
    parser, args = parse_args()  # Your argparse arguments
    save_args_as_config(args, parser=parser)
"""

import json
import os
import sys


def _find_output_dir(args_dict):
    """Find output directory from various possible argument names."""
    possible_names = ['output_dir', 'output', 'out', 'output_path', 'output_directory']
    for name in possible_names:
        if name in args_dict and args_dict[name] is not None:
            return args_dict[name]
    return None


def save_args_as_config(args, output_dir=None, parser=None, save_full_config=False):
    """
    Save argparse arguments to config files in the output directory.
    
    By default, saves only user-specified arguments to config.json.
    Optionally saves all arguments (including defaults) to config_full.json.
    
    This function is agnostic to specific argument names and automatically
    detects common variations of input/output path arguments.
    
    Args:
        args: argparse.Namespace object containing parsed command-line arguments.
              Must have an output directory attribute (output_dir, output, out, etc.)
        output_dir: Optional output directory path (overrides auto-detection)
        parser: Optional argparse.ArgumentParser to detect which args were explicitly provided
        save_full_config: If True, also save config_full.json with all arguments (default: False)
    
    Raises:
        AttributeError: If output directory cannot be determined from args.
    """
    # Convert args namespace to dictionary
    args_dict = vars(args)
    
    # Get output directory
    if output_dir is None:
        output_dir = _find_output_dir(args_dict)
        if output_dir is None:
            raise AttributeError(
                "Could not find output directory in args. "
                "Expected one of: output_dir, output, out, output_path. "
                "Or provide output_dir explicitly: save_args_as_config(args, output_dir='path/to/dir')"
            )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If parser is provided, extract explicitly provided arguments from command line
    if parser is not None:
        user_args = {}
        
        # Get command line arguments (excluding the script name)
        cmd_args = sys.argv[1:]
        
        # Build a mapping of argument names to their destinations
        arg_to_dest = {}
        for action in parser._actions:
            if action.dest == 'help':
                continue
            for option_string in action.option_strings:
                arg_to_dest[option_string] = action.dest
        
        # Track which destinations were explicitly provided
        provided_dests = set()
        i = 0
        while i < len(cmd_args):
            arg = cmd_args[i]
            
            # Check if this is an option (starts with -)
            if arg.startswith('-'):
                # Handle --arg=value format
                if '=' in arg:
                    option = arg.split('=')[0]
                    if option in arg_to_dest:
                        provided_dests.add(arg_to_dest[option])
                # Handle --arg value format or --flag format
                else:
                    if arg in arg_to_dest:
                        dest = arg_to_dest[arg]
                        provided_dests.add(dest)
                        # Check if next item is the value (not another flag)
                        if i + 1 < len(cmd_args) and not cmd_args[i + 1].startswith('-'):
                            i += 1  # Skip the value in next iteration
            i += 1
        
        # Build user_args dict with only explicitly provided arguments
        for dest in provided_dests:
            if dest in args_dict:
                user_args[dest] = args_dict[dest]
        
        # Save user-specified args as config.json
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(user_args, f, indent=4, sort_keys=True)
        
        print(f"Configuration saved to: {config_path}")
        
        # Optionally save full config
        if save_full_config:
            config_full_path = os.path.join(output_dir, "config_full.json")
            with open(config_full_path, 'w') as f:
                json.dump(args_dict, f, indent=4, sort_keys=True)
            print(f"Full configuration saved to: {config_full_path}")
    else:
        # Without parser, just save all args as config.json
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(args_dict, f, indent=4, sort_keys=True)
        print(f"Configuration saved to: {config_path}")

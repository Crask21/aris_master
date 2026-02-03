"""
Example usage of the ConfigLoader.
"""
from src.config_loader.config_loader_json import ConfigLoaderJSON


def main():
    # Example 1: Load configuration
    config = ConfigLoaderJSON('config/diffusion_model.json')
    
    # Example 2: Get entire config
    print("Entire config:")
    print(config.get())
    print()
    
    # Example 3: Get specific values using dot notation
    print("Batch size:", config.get('data.batch_size'))
    print("Learning rate:", config.get('data.learning_rate'))
    print()
    
    # Example 3b: Get values using dictionary-style bracket notation
    print("Batch size (bracket notation):", config['data']['batch_size'])
    print("Learning rate (bracket notation):", config['data']['learning_rate'])
    print("Entire data section:", config['data'])
    print()
    
    # Example 4: Get entire section
    data_config = config.get_section('data')
    print("Data section:", data_config)
    print()
    
    # Example 5: Set a value (dot notation)
    config.set('data.batch_size', 32)
    print("Updated batch size:", config.get('data.batch_size'))
    
    # Example 5b: Set a value (bracket notation)
    config['data']['batch_size'] = 64
    print("Updated batch size (bracket):", config['data']['batch_size'])
    print()
    
    # Example 6: Check membership with 'in' operator
    print("'data' in config:", 'data' in config)
    print("'missing' in config:", 'missing' in config)
    print()
    
    # Example 7: Update multiple values
    config.update({
        'data.epochs': 300,
        'data.learning_rate': 0.0002
    })
    print("Updated epochs:", config.get('data.epochs'))
    print("Updated learning rate:", config.get('data.learning_rate'))
    print()
    
    # Example 8: Add new section
    config.set('training.optimizer', 'adam')
    config.set('training.scheduler', 'cosine')
    print("New training section:", config.get_section('training'))
    print()
    
    # Example 9: Check if key exists
    print("Has 'data.batch_size':", config.has('data.batch_size'))
    print("Has 'missing.key':", config.has('missing.key'))
    print()
    
    # Example 10: Save to new file (optional)
    # config.save('config/diffusion_model_updated.yaml')
    
    # Example 11: Alternative convenience function


if __name__ == '__main__':
    main()

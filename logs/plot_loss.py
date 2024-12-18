import matplotlib.pyplot as plt
import json

def main():
    config_path = './config.json'
    with open(config_path) as f:
        config = json.load(f)
    
    
    
    x = range(config['epochs'])
    y = config['loss']
    # print(json.dumps(config, indent=4))

    plt.plot(x, y)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('./loss.png')

if __name__ == '__main__':
    main()
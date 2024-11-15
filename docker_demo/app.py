# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO
import torch
import matplotlib.pyplot as plt
from src.count_primes import count_primes

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

def background_task():
    if torch.cuda.is_available():
        socketio.emit('log', {'message': 'CUDA is available'})
    else:
        socketio.emit('log', {'message': 'CUDA is not available'})

    # Perform large scale computation
    x = torch.rand(1000, 1000)
    y = torch.rand(1000, 1000)
    z = torch.matmul(x, y)
    socketio.emit('log', {'message': 'Matrix multiplication completed.'})

    # Count the number of prime numbers up to 1 million
    N = 1_000_000
    x_values = range(N)
    y_values = [0] * N

    for i in range(N):
        y_values[i] = count_primes(i)
        # Emit progress every 10,000 iterations
        if i % 10000 == 0:
            progress = (i / N) * 100
            socketio.emit('progress', {'progress': progress})
    # Plotting
    plt.plot(x_values, y_values)
    plt.xlabel("N")
    plt.ylabel("Primes")
    plt.title("Prime distribution")
    plt.savefig("primes.png")
    socketio.emit('log', {'message': 'Computation and plotting done!'})

@socketio.on('connect')
def handle_connect():
    socketio.emit('log', {'message': 'Client connected'})
    socketio.start_background_task(target=background_task)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0')

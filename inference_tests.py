from camera_interface.OAK_D_API import OAK_D

import time
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import model_zoo
torch.backends.quantized.engine = 'qnnpack'

# normilization between 0 and 1
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640)),
    transforms.Normalize(mean=[0.5], std=[0.5])    
])

def run_tests():
    # pipeline for interacting with OAK-D camera
    oak_d = OAK_D()
    iter = 300

    # iterate though all models in model_zoo
    for model_name, model in model_zoo.model_zoo.items():
        print(f"Testing {model_name}")
        
        last_logged = time.time()
        frame_count = 0
        fps = []
        with torch.no_grad():
            for i in range(iter):
                color_frame = oak_d.get_color_frame(show_fps=True)

                # preprocess
                input_tensor = preprocess(color_frame)

                # create a mini-batch as expected by the model
                input_batch = input_tensor.unsqueeze(0)

                # run model
                output = model(input_batch)

                # print model performance
                frame_count += 1
                now = time.time()
                if now - last_logged > 1:
                    current_fps = frame_count / (now-last_logged)
                    print(f"{current_fps} fps")
                    last_logged = now
                    frame_count = 0
                    fps.append(current_fps)
        
        # logging fps results into a file
        with open(f"results/{model_name}.txt", "w") as f:
            f.write(f"{model_name}\n")
            for fps_val in fps[2:]:
                f.write(f"{fps_val}\n")

def plot_results():
    for model_name in model_zoo.model_zoo.keys():
        with open(f"results/{model_name}.txt", "r") as f:
            lines = f.readlines()
            lines = [float(line) for line in lines[1:]]
            plt.plot(lines, label=model_name)
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("FPS")
    plt.show()

    


if __name__ == "__main__":
    run_tests()
    # plot_results()
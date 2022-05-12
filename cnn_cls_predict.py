import numpy as np
import torch
import torch.nn.functional as F
import model_cnn_cls

class model_cnn():
    def __init__(self, model=None, param_path=None):
        self.model = model
        checkpoint = torch.load(param_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def output(self, input):
        input = np.expand_dims(input,axis=0)
        input = torch.from_numpy(input).float()
        with torch.no_grad():
            probs = torch.squeeze(F.softmax(self.model(input), dim=1)).numpy()

        return probs

    def output_max(self, input):
        input = np.expand_dims(input,axis=0)
        input = torch.from_numpy(input).float()
        with torch.no_grad():
            output = np.argmax(torch.squeeze(F.softmax(self.model(input), dim=1)).numpy())

        return output

if __name__ == "__main__":
    input = np.random.randn(3,300)
    print(input)
    model_cnn1 = model_cnn(model=model_cnn_cls.CNN3(), param_path="train_data_pga80_snr5_update_300_label4_model.ckpt")

    print("Probability:", model_cnn1.output(input))
    print("Predict:", model_cnn1.output_max(input))


import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.current_device())
with torch.no_grad():
    state_dict = torch.load(
        r"/media/yujiajing0408/Data/Uploads/Weights/DenoisingMamba/4SPP/Step1/checkpoints/last.ckpt")
    torch.save(state_dict,r"/media/yujiajing0408/Data/Uploads/Weights/DenoisingMamba/4SPP/Step1/checkpoints/last_wo_grad.ckpt")

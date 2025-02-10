import torch
import numpy as np
from sklearn.metrics import average_precision_score


if __name__ == '__main__':

    model_teacher = TeacherModel()
    model_teacher.load_state_dict(torch.load('model/ALGCN_mirflickr.pth'))
    model_teacher = model_teacher.cuda()


    test_data_loader, input_data_par = get_loader('data/mirflickr/', batch_size=100)


    test(model_teacher, test_data_loader, input_data_par)

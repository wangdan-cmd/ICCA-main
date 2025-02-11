import os

import torch
import torch.optim as optim
import numpy as np
from scipy.io import loadmat

from evaluate import fx_calc_map_label


if __name__ == '__main__':

    dataset = 'mirflickr'
    embedding = 'none'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    DATA_DIR = 'data/' + dataset + '/'
    EVAL = False


    if dataset == 'mirflickr':
        alpha = 0.5
          beta=0.8
        gamma = 0.2
         eta=0.5
        lambda=0.6
        MAX_EPOCH = 30
        batch_size = 100
        lr = 5e-5
    elif dataset == 'NUS-WIDE':
        alpha = 0.5
          beta=0.8
        gamma = 0.2
         eta=0.5
        lambda=0.6
        batch_size = 1024
        lr = 5e-5
       elif dataset == 'MS-COCO':
         alpha = 0.5
          beta=0.8
        gamma = 0.2
         eta=0.5
        lambda=0.6
        batch_size = 1024
        lr = 5e-5
       
    else:
        raise NameError("Invalid dataset name!")

    seed = 103
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    if embedding == 'glove':
        inp = loadmat('embedding/' + dataset + '-inp-glove6B.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'googlenews':
        inp = loadmat('embedding/' + dataset + '-inp-googlenews.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'fasttext':
        inp = loadmat('embedding/' + dataset + '-inp-fasttext.mat')['inp']
        inp = torch.FloatTensor(inp)
    else:
        inp = None


    print('...Data loading is beginning...')
    data_loader, input_data_par = get_loader(DATA_DIR, batch_size)
    print('...Data loading is completed...')


    model_teacher = TeacherModel()
    model_student = StudentModel()
    classifier_teacher = Classifier()
    classifier_student = Classifier()


    params_to_update = model_teacher.get_config_optim(lr)
    optimizer_teacher = optim.Adam(params_to_update, lr=lr, betas=betas)
    optimizer_student = optim.Adam(model_student.parameters(), lr=lr, betas=betas)


    if EVAL:
        model_teacher.load_state_dict(torch.load('model/ICCA_' + dataset + '.pth'))
    else:
        print('...Training is beginning...')
        model_teacher, img_acc_hist, txt_acc_hist, loss_hist = train(model_teacher, model_student, classifier_teacher, classifier_student, data_loader, optimizer_teacher, optimizer_student, alpha, gamma, t)
        print('...Training is completed...')
        torch.save(model_teacher.state_dict(), 'model/ICCA_' + dataset + '.pth')


    print('...Evaluation on testing data...')
    view1_feature, view2_feature, view1_predict, view2_predict, classifiers, _ = model_teacher(
        torch.tensor(input_data_par['img_test']).cuda(), torch.tensor(input_data_par['text_test']).cuda())
    label = input_data_par['label_test']
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()

    img_to_txt = fx_calc_map_label(view1_feature, view2_feature, label)
    print('...Image to Text MAP = {}'.format(img_to_txt))

    txt_to_img = fx_calc_map_label(view2_feature, view1_feature, label)
    print('...Text to Image MAP = {}'.format(txt_to_img))

    print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))



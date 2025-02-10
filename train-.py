print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
def semantic_difference_loss(image_enhanced, text_enhanced, sample_representations):

    image_diff = F.mse_loss(image_enhanced, sample_representations, reduction='none')


    text_diff = F.mse_loss(text_enhanced, sample_representations, reduction='none')


    cross_modal_diff = F.mse_loss(image_enhanced, text_enhanced, reduction='none')


    loss = torch.sum(image_diff - text_diff + cross_modal_diff)

    def semantic_consistency_loss(z_iv, z_tj, r_i, temperature=1.0):

        cosine_sim = F.cosine_similarity(z_iv.unsqueeze(1), z_tj.unsqueeze(0), dim=-1)


        S_ij = r_i.unsqueeze(1) * cosine_sim

        loss = torch.sum(torch.log(1 + torch.exp(S_ij)) * temperature)

        def semantic_matching_loss(z_iv, z_tj, z_ml_iv, z_ml_tj, temperature=1.0):

            cosine_sim_iv = F.cosine_similarity(z_iv.unsqueeze(1), z_tj.unsqueeze(0), dim=-1)
            cosine_sim_ml = F.cosine_similarity(z_ml_iv.unsqueeze(1), z_ml_tj.unsqueeze(0), dim=-1)

            w_ij = 0.5 * (cosine_sim_iv + cosine_sim_ml)

            loss = torch.sum(torch.exp(- w_ij) * temperature)

def classification_loss(p, y):

    return F.binary_cross_entropy(p, y)

def train(model_teacher, model_student, classifier_teacher, classifier_student, data_loader, optimizer_teacher, optimizer_student, beta, gamma, eta):
    model_teacher.train()
    model_student.train()
    total_loss = 0

    for data in data_loader:

        image_features, text_features, image_labels, text_labels, multi_labels = data


        z_iv, z_it, z_iml = model_teacher(image_features, text_features)


        z_iv_student, z_it_student = model_student(image_features, text_features)

        loss_teacher = teacher_loss(z_iv, z_it, z_iml, image_labels, text_labels, multi_labels, classifier_teacher, beta, gamma, eta)


        loss_student = student_loss(z_iv, z_it, z_iml, classifier_teacher, classifier_student)




        optimizer_teacher.zero_grad()
        optimizer_student.zero_grad()
        total_loss.backward()


        optimizer_teacher.step()
        optimizer_student.step()

    return total_loss.item()



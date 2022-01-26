usage: main.py [-h] [--type TYPE] [--update_scale UPDATE_SCALE] [--ds DS] [--half HALF] [--save_model SAVE_MODEL]
               [--save_path SAVE_PATH]




To train the model:

python3 main.py --type multi-agent --save_model true --half false --update_scale 0.2 --ds new_ds --save_path output_model/output.pt


To get training dataset:
https://drive.google.com/u/0/uc?id=11ye00sHFY5re2NOBRKreg-tVbDNrc7Xd&export=download
with about 2600 samples


To use the model:
1. run python dataHelper.py to build .pkl file for each user (Error might be caused due to the main function is modified to adapt projectx dataset)
2. run python main.py --type multi-agent --save_model true --half false --update_scale 0.2 --ds new_ds --save_path output_model/output.pt
   update_scale subject to change.





To use the model in projectX:

1.Dataset pareperation: we need a dataset folder named "projectx", which contains the sub-floders "positive" and "negative". In the these two sub-folders, folders of each users is named with their username. Within each user folder, we need all their images(.jpg) and timeline(.txt) of tweets. Format refered to the readme.md in new_ds.

2.Build the .pkl for our collected user. Just run 
	
	python dataHelper.py

3.To use the model. The trained model is stored in output_model/output.pt . To use it, in our program import all .py files in this folder and run 
	
	ds = get_type_dataset(load_pickle=True, build_pickle=False, need_image=True, ds="projectx", half=half)
    	project_loader = DataLoader(ds.ds, batch_size=1, collate_fn=collate, shuffle=True)


	model = torch.load('output_model/output.pt')
        model.textEncoder.encoder.flatten_parameters()

	loss = 0
        correct = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        with torch.no_grad():
            for user, label in project_loader:
                cf_loss, prediction = model.update_buffer(user[0], label,
                                                          need_backward=False,
                                                          train_classifier=False,
                                                          update_buffer=False)
                loss += cf_loss
                if prediction == label[0]:
                    correct += 1
                    if prediction == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if prediction == 1:
                        fp += 1
                    else:
                        fn += 1
        dev_accuracy = correct / len(ds.dev_ds)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        n_precision = tn / (tn + fn + eps)
        n_recall = tn / (tn + fp + eps)
        macro_P = (precision + n_precision) / 2
        macro_R = (recall + n_recall) / 2
        F1 = 2 * precision * recall / (precision + recall + eps)
        n_F1 = 2 * n_precision * n_recall / (n_precision + n_recall + eps)
        macro_F1 = (F1 + n_F1) / 2
        print(
            f'ProjectX dataset: Acc {dev_accuracy}\tprecision {macro_P}\trecall {macro_R}\tF1 {macro_F1}\tloss {loss}')

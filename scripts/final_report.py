from sklearn.metrics import classification_report,accuracy_score,f1_score,confusion_matrix
from seaborn import heatmap
import matplotlib.pyplot as plt
import os

class Final_Report:
    def __init__(self,output_path,all_labels,all_pred,for_group):

        self.output_path=output_path
        self.all_labels,self.all_pred=all_labels,all_pred
        #the same dict used in encoding reversed 
        team_action_dct = {
            0:'l-pass',     1:'r-pass',
            2:'l-spike',    3:'r_spike',
            4:'l_set',      5:'r_set',
            6:'l_winpoint', 7:'r_winpoint'
        }
        player_action_dct = {
            0:'waiting', 1:'setting', 2:'digging', 
            3:'falling', 4:'spiking', 5:'blocking',
            6:'jumping', 7:'moving', 8:'standing'
        }
        sorted_keys_group = sorted(team_action_dct.keys())
        sorted_keys_player = sorted(player_action_dct.keys())
        if for_group:
            self.class_names = [team_action_dct[i] for i in sorted_keys_group]
            self.sorted_keys = sorted_keys_group
        else:
            self.class_names = [player_action_dct[i] for i in sorted_keys_player]
            self.sorted_keys = sorted_keys_player

    def creat_report(self):
        #pass labels and target names to make sure that every class mapped for it's correct calss name
        cla_rep = classification_report(self.all_labels,self.all_pred,labels=self.sorted_keys,target_names=self.class_names)
        with open(os.path.join(self.output_path,'report.txt'),'w') as f:
            f.write("                         --- Classification Report ---\n")
            f.write(cla_rep)
            f.write('\n\n'+'-'*100+'\n')

            f.write("                         --- Global Metrics ---   \n")
            f.write(f'Accuracy:{accuracy_score(self.all_labels,self.all_pred)*100:.2f} %')
            f.write('\n')
            f.write(f'F1_Score:{f1_score(self.all_labels,self.all_pred,average='weighted')*100:.2f} %')

    def create_confusion_matrix(self):
        # pass labels to make sure that every class appear in the matrix even if not exits in the labels and pred
        conv_word = confusion_matrix(self.all_labels, self.all_pred, normalize='true',labels=self.sorted_keys)
        # Plot the heatmap
        plt.figure(figsize=(10, 8)) 
        heatmap(
            conv_word,
            annot=True, 
            fmt=".2f",  
            cmap="Blues", 
            # replace labels(0,1,2,..) with class names
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        # labelpad: adds distance between the axis and the label
        plt.xlabel('Predicted labels', labelpad=20, fontsize=12)
        plt.ylabel('True labels', labelpad=20, fontsize=12)

        # pad: adds distance between the title and the top of the heatmap
        plt.title(f'{self.output_path.split('/')[-1]} Confusion Matrix ', pad=30, fontsize=14)

        # Ensures labels aren't cut off after adding padding
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path,'confusion_matrix.png'))
        plt.close()            

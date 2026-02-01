from sklearn.metrics import classification_report,accuracy_score,f1_score,confusion_matrix
from seaborn import heatmap
import matplotlib.pyplot as plt

class Final_Report:
    def __init__(self,baseline,all_labels,all_pred):
        self.baseline=baseline
        self.all_labels,self.all_pred=all_labels,all_pred
        self.team_action_dct = {
        0:'l_pass',     1:'r_pass',
        2:'l_spike',    3:'r_spike',
        4:'l_set',      5:'r_set',
        6:'l_winpoint', 7:'r_winpoint'
        }
        self.class_names = [self.team_action_dct[i] for i in range(len(self.team_action_dct))]
    def creat_report(self):
        cla_rep = classification_report(self.all_labels,self.all_pred,labels=self.class_names)
        with open(f'outputs/{self.baseline.upper()}/{self.baseline}_report.txt','w') as f:
            f.write("                         --- Classification Report ---\n")
            f.write(cla_rep)
            f.write('\n\n'+'-'*100+'\n')

            f.write("                         --- Global Metrics ---   \n")
            f.write(f'Accuracy:{accuracy_score(self.all_labels,self.all_pred):.2f}')
            f.write('\n')
            f.write(f'F1_Score:{f1_score(self.all_labels,self.all_pred,average='weighted'):.2f}')

    def create_confusion_matrix(self):
        conv_word = confusion_matrix(self.all_labels, self.all_pred, normalize='all')
        # Plot the heatmap
        plt.figure(figsize=(10, 8)) 
        heatmap(
            conv_word,
            annot=True, 
            fmt=".2f",  
            cmap="Blues", 
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        # labelpad: adds distance between the axis and the label
        plt.xlabel('Predicted labels', labelpad=20, fontsize=12)
        plt.ylabel('True labels', labelpad=20, fontsize=12)

        # pad: adds distance between the title and the top of the heatmap
        plt.title(f'Confusion Matrix For Baseline {self.baseline.upper()}', pad=30, fontsize=14)

        # Ensures labels aren't cut off after adding padding
        plt.tight_layout()
        plt.savefig(f'outputs/{self.baseline.upper()}/{self.baseline}_confusion_matrix.png')
            

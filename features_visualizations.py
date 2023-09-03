import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visual:
    def __init__(self, dataset):
        self.dataset = dataset

        self.quant_feat = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_sodium', 'serum_creatinine', 'time']
        self.cat_feat = ["anaemia","diabetes","high_blood_pressure","sex","smoking","DEATH_EVENT"]

    def countplot(self):

        target_count = self.dataset.DEATH_EVENT.value_counts()
        death_color = ['darkgreen', 'red']
        with plt.style.context('ggplot'):
            plt.figure(figsize=(6, 5))
            sns.countplot(data=self.dataset, x='DEATH_EVENT', palette=death_color)
            for name , val in zip(target_count.index, target_count.values):
                plt.text(name, val/2, f'{round(val/sum(target_count)*100, 2)}%\n({val})', ha='center',
                color='white', fontdict={'fontsize':13})
            plt.xticks(ticks=target_count.index, labels=['No', 'True'])
            plt.yticks(np.arange(0, 230, 25))
            plt.show()
        
    def age_distribution(self):
        f,ax = plt.subplots(figsize=(6,6))
        sns.distplot(self.dataset["age"],kde=True,axlabel="Age")
        ax.text(0.5,0.9,"Mean -" + str("{:.2f}".format(self.dataset["age"].mean())),transform=ax.transAxes)
        ax.text(0.5,0.85,"Standard Deviation -" + str("{:.2f}".format(self.dataset["age"].std())),transform=ax.transAxes)
        ax.set_title("Age Distribution")
        plt.show()

    def age_boxplot(self):
        f,ax = plt.subplots(figsize=(6,6))
        sns.boxplot(x="DEATH_EVENT",y="age",data=self.dataset)
        ax.set_title("Age by Death Event")
        ax.set_xlabel("Death Event (0-No, 1-Yes)")
        ax.set_ylabel("Age")
        plt.show()
    
    def serum_boxplot(self):
        f,ax = plt.subplots(figsize=(6,6))
        sns.boxplot(x="DEATH_EVENT",y="serum_sodium",data=self.dataset)
        ax.set_title("Serum Sodium by Death Event")
        ax.set_xlabel("Death Event (0-No, 1-Yes)")
        ax.set_ylabel("Serum Sodium")
        plt.show()
    
    def time_boxplot(self):
        f,ax = plt.subplots(figsize=(6,6))
        sns.boxplot(x="DEATH_EVENT",y="time",data=self.dataset)
        ax.set_title("Time by Death Event")
        ax.set_xlabel("Death Event (0-No, 1-Yes)")
        ax.set_ylabel("Time")
        plt.show()

    def ejection_boxplot(self):
        f,ax = plt.subplots(figsize=(6,6))
        sns.boxplot(x="DEATH_EVENT",y="ejection_fraction",data=self.dataset)
        ax.set_title("Ejection Fraction by Death Event")
        ax.set_xlabel("Death Event (0-No, 1-Yes)")
        ax.set_ylabel("Ejection Fraction")
        plt.show()
    
    def serum_creatinine_boxplot(self):
        f,ax = plt.subplots(figsize=(6,6))
        sns.boxplot(x="DEATH_EVENT",y="serum_creatinine",data=self.dataset)
        ax.set_title("Serum Creatinine by Death Event")
        ax.set_xlabel("Death Event (0-No, 1-Yes)")
        ax.set_ylabel("Serum Creatinine")
        plt.show()
    
    def phos_creatinine_boxplot(self):
        f,ax = plt.subplots(figsize=(6,6))
        sns.boxplot(x="DEATH_EVENT",y="creatinine_phosphokinase",data=self.dataset)
        ax.set_title("Creatinine phosphokinase by Death Event")
        ax.set_xlabel("Death Event (0-No, 1-Yes)")
        ax.set_ylabel("Creatinine Phosphokinase")
        plt.show()

    def platelets(self):
        f,ax = plt.subplots(figsize=(6,6))
        sns.boxplot(x="DEATH_EVENT",y="platelets",data=self.dataset)
        ax.set_title("Platelets by Death Event")
        ax.set_xlabel("Death Event (0-No, 1-Yes)")
        ax.set_ylabel("Platelets")
        plt.show()   

    def heatmap(self):
        corr = self.dataset.corr()
        ax, fig = plt.subplots(figsize=(10,10))
        sns.heatmap(corr, vmin=-1, cmap='coolwarm', annot=True)
        plt.show()

    def histogram_var(self):
        colors = sns.color_palette("dark")
        with plt.style.context('ggplot'):
            plt.figure(figsize=(10, 10))
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            for i, (col, name) in enumerate(zip(colors, self.quant_feat)):
                plt.subplot(3, 3, i+1)
                sns.histplot(data=self.dataset, x=name, color=col)
            plt.suptitle('Histogram of Quantitative Variables', fontsize=15)
        plt.show()
    
    def pairplot(self):
        sns.pairplot(self.dataset[["age", "creatinine_phosphokinase", "ejection_fraction", "serum_creatinine", "time", "platelets", "serum_sodium"]], diag_kind="kde");
        plt.show()

    def barplot_cat(self):
        r = c = 0
        fig,ax = plt.subplots(3,2,figsize=(14,12))
        for n,i in enumerate(self.cat_feat[:-1]):
            ct = pd.crosstab(columns=self.dataset[i],index=self.dataset["DEATH_EVENT"],normalize="columns")
            ct.T.plot(kind="bar",stacked=True,color=["green","red"],ax=ax[r,c])
            ax[r,c].set_ylabel("% of observations")
            c+=1
            if (n+1)%2==0:
                r+=1
                c=0
        ax[r,c].axis("off")
        plt.show()



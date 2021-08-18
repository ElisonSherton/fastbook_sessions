import torch
import numpy as np
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns

class utils:   
    @classmethod
    def getMetrics(self, p, t, thresh):
        """
        Given the predictions and ground truth values for one attribute/class, computes the common metrics
        for the given threshold and returns them all
        """
        # Get the ypreds in binary format
        yp = (p >= thresh) * 1
        
        # Compute the metrics
        pr = precision_score(t, yp)
        rc = recall_score(t, yp, pos_label = 1)
        tnr = recall_score(t, yp, pos_label = 0)
        fpr = 1 - tnr
        acc = accuracy_score(t, yp)
        f1 = f1_score(t, yp)

        return [pr, rc, fpr, acc, f1]

    @classmethod
    def closestPoint(self, df, x = "FPR", y = "Recall_TPR", acc = "Accuracy", thr = "Threshold"):
        """
        Given a dataframe of metrics, figures out the point on ROC-AUC curve which is closest to the top 
        left corner i.e. (0, 1) of the curve
        """
        # Define a list of all the points
        points = [(x, y) for x, y in zip(df[x], df[y])]

        # Function to compute the Euclidean distance
        ED = lambda point: np.sqrt((point[0] - 0) ** 2 + (point[1] - 1) ** 2)

        # Get all the distances
        distances = np.array([ED(x) for x in points])

        # Get the idx of the shortest distance point
        closest = distances.argmin()

        # Get the FPR, TPR point corresponding to the shortest distance
        closestPoint = [df[x].iloc[closest].item(), df[y].iloc[closest].item()]

        return closestPoint, df[acc].iloc[closest].item(), df[thr].iloc[closest].item()
    
    @classmethod
    def getSummary(self, predictions, targets, plot = True):
        """
        Given a set of predictions and targets, computes a best threshold based on closest point to the y-axis 
        and the best f1-score metrics

        Optionally plots the metrics
        """

        # Create a container to hold all the metrics
        summary = []

        # Create a list of thresholds evenly spaced between 0 to 1
        thresholds = np.arange(0, 1, 0.01)

        # For all the thresholds, compute the metrics
        for t in thresholds:
            metrics = self.getMetrics(predictions, targets, t)
            summary.append([t] + metrics)

        # Organize the thresholds & metrics in a dataframe
        df = pd.DataFrame(summary, columns = ["Threshold", "Precision", "Recall_TPR", "FPR", "Accuracy", "f1Score"])
        closestPt, closestAcc, closestThr = self.closestPoint(df)

        # Figure out the point at which f1-score is maximum 
        bestf1 = df[df["f1Score"] == df["f1Score"].max()].reset_index(drop = True).iloc[0, :]
        bestf1x, bestf1y = bestf1["FPR"].item(), bestf1["Recall_TPR"].item()

        # Find out the point where accuracy is maximum
        bestAcc = df[df["Accuracy"] == df["Accuracy"].max()].reset_index(drop = True).iloc[0, :]
        bestAccx, bestAccy = bestAcc["FPR"].item(), bestAcc["Recall_TPR"].item()

        # Plot the metrics obtained from above
        if plot:
            fig, ax = plt.subplots(1, 1, figsize = (6, 4))
            sns.lineplot(data = df, x = "FPR", y = "Recall_TPR", ax = ax)
            ax.scatter(*closestPt, c = "red", label = "Closest Point")
            ax.scatter(bestf1x, bestf1y, c = "green", label = "Best F1-score")
            ax.scatter(bestAccx, bestAccy, c = "orange", label = "Best Accuracy Point")
            ax.legend()
            fig.suptitle("ROC-AUC Curve")
            fig.tight_layout();

        return {"bestF1_Point": [bestf1["Threshold"].item(), bestf1["Accuracy"].item()], 
                "bestAcc_Point": [bestAcc["Threshold"].item(), bestAcc["Accuracy"].item()], 
                "closest_Point": [closestThr, closestAcc]}
    
    @classmethod
    def getThresholdsByClass(self, predictions, targets, classNames):
        """
        Given a set of predictions, targets and classNames, checks the ROC-AUC Curve and returns 
        threshold for best metric by three strategies

        1. Best f1-score point
        2. Best Accuracy point
        3. Closest to top left point (0, 1) in ROC-AUC Curve 

        """
        thresholds = {}
        for classNumber in range(len(classNames)):
            className = classNames[classNumber]

            preds = predictions[:, classNumber].numpy()
            targs = targets[:, classNumber].numpy().astype(np.int)

            thresholds[className] = self.getSummary(preds, targs, False)

        return thresholds
    
    @classmethod
    def getFinalAccuracy(self, predictions, targets, classNames, strategy = "Accuracy"):
    
        # Get the thresholds for each class as per the strategy
        thrs = self.getThresholdsByClass(predictions, targets, classNames)
        if strategy == "Accuracy":
            thrs = [v["bestAcc_Point"][0] for v in thrs.values()]
        elif strategy == "f1Score":
            thrs = [v["bestF1_Point"][0] for v in thrs.values()]
        elif strategy == "Closest":
            thrs = [v["closest_Point"][0] for v in thrs.values()]

        # Get the predictons array and compute the accuracy
        predArray = []
        for idx in range(predictions.shape[1]):
            p = ((predictions[:, idx] > thrs[idx]) * 1).unsqueeze(1)
            predArray.append(p)
        ypreds = torch.cat(predArray, axis = 1)

        return (ypreds == targets).float().mean().item()
    
    @classmethod
    def refactorPredictions(self, predictions):
        
        # Define a container for holding the predictions
        predictionsArray = []
        
        # Get all the predictions by indexing into individual fc Clasifier layers
        for pred in predictions:
            sm_pred = pred.softmax(dim = 1)[:, 1]
            predictionsArray.append(sm_pred)
        
        # Stack all the predictions together
        predictions = torch.stack(predictionsArray, dim = 1)
        
        return predictions
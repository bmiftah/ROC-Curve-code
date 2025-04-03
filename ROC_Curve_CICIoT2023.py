### Code snippt to plot ROC Curve for CICIoT2023 dataset for SAT-BILSTM Model


## required imports goes here

### Roc-CURVE

# load the history and trained model

history_filepath = '/content/drive/MyDrive/IDS-Research/CICIoT2023_SAT_BI_LSTM_December_14_history.npz'
model_checkpoint_filepath = '/content/drive/MyDrive/IDS-Research/BI-LSTM_Attention_CICIoT2023_0814.h5'
#### Load test data and make predictions

# Load the datasets using the correct filenames
X_train = np.load(os.path.join(save_dir, 'X_train.npy'))
X_test = np.load(os.path.join(save_dir, 'X_test.npy'))
Y_train = np.load(os.path.join(save_dir, 'y_train.npy'))
Y_test = np.load(os.path.join(save_dir, 'y_test.npy'))
X_test = X_test
y_test = Y_test

# Create the save directory
os.makedirs(save_dir, exist_ok=True)

# Reload the model
loaded_model = load_model(model_checkpoint_filepath)
y_pred_probs = loaded_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

#  Binarize the labels for multiclass classification

y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_binarized.shape[1]

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute ROC curve and AUC for micro-average
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute ROC curve and AUC for macro-average
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot ROC curves for each class, micro-average, and macro-average
plt.figure(figsize=(9.0, 6.0), dpi=300)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

# Plot micro-average and macro-average curves
plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})', color='deeppink', linestyle=':', linewidth=4)
plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})', color='navy', linestyle=':', linewidth=4)

# Add reference line and labels
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.xticks(np.arange(0.0, 1.1, step=0.2), fontsize=20)  # Set x-ticks
plt.yticks(np.arange(0.0, 1.1, step=0.2), fontsize=20)  # Set y-ticks

# Adjust the legend position to avoid overlap
plt.legend(loc="lower right", fontsize=14)

# Save and show the plot
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ROC_Curve_with_Micro_Macro_averages.jpeg'), dpi=300, bbox_inches='tight')
plt.show()
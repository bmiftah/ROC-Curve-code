### Code snippt to plot ROC Curve for CICIoT2023 dataset for SAT-BILSTM Model

### This is code is for reference but it won't run as it stands here( paths ,model , dataset and libraries  etc are required for running it)

## required imports goes here

# Load the model and data
checkpoint_path = '/content/drive/MyDrive/IDS-Research/Bi_LSTM_Attention_December_25.h5'
bilstm_attention_model = load_model(checkpoint_path)

history_path = '/content/drive/MyDrive/IDS-Research/Bi_LSTM_Attention_MDcic2023_history_0725.npz'

# Load test data
X_test1 = np.load(os.path.join(save_dir, 'X_test1.npy'))
Y_test = np.load(os.path.join(save_dir, 'Y_test.npy'))

# Make predictions
y_pred_probs = bilstm_attention_model.predict(X_test1)
y_pred = np.argmax(y_pred_probs, axis=1)

# ROC curve for each class
Y_test_binarized = label_binarize(Y_test, classes=np.unique(Y_test))
n_classes = Y_test_binarized.shape[1]
fpr, tpr, roc_auc = {}, {}, {}

# Calculate FPR, TPR, and AUC
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test_binarized[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute ROC Curve for Micro-average and Macro-average
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test_binarized.ravel(), y_pred_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(8.0, 5.0), dpi=300)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {["BENIGN", "Syn", "NetBIOS", "UDP", "MSSQL", "Portmap", "LDAP", "UDPLag"][i]} (area = {roc_auc[i]:.4f})')
plt.plot(fpr["micro"], tpr["micro"], label=f'micro-average ROC curve (area = {roc_auc["micro"]:.4f})',
         color='deeppink', linestyle=':', linewidth=4)
plt.plot(fpr["macro"], tpr["macro"], label=f'macro-average ROC curve (area = {roc_auc["macro"]:.4f})',
         color='navy', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# Adjusting the legend further to make it smaller
plt.legend(loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(plot_directory, 'ROC_Curves_2019_SP_1106.jpeg'), format='jpeg', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(plot_directory, 'ROC_Curves_2019_SP_1106.svg'), format='svg', dpi=300, bbox_inches='tight')
plt.show()

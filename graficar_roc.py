import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from configs.configs import get_configs_shanghai
from data.test_dataset import AbnormalDatasetGradientsTest
from model.model_factory import mae_cvt_patch8
from util import misc
from util.abnormal_utils import filt

def main():
    # 1. Configurar parametros
    args = get_configs_shanghai()
    args.output_dir = "experiments/shanghai"
    device = torch.device(args.device)

    # 2. Preparar el DataLoader de validación
    dataset_test = AbnormalDatasetGradientsTest(args)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # 3. Instanciar el modelo
    model = mae_cvt_patch8(
        norm_pix_loss=args.norm_pix_loss, 
        img_size=args.input_size,
        use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
        abnormal_score_func=args.abnormal_score_func,
        masking_method=args.masking_method,
        grad_weighted_loss=args.grad_weighted_rec_loss
    ).float()
    
    # Asegurar que está en modo Teacher-Student
    model.train_TS = True
    model.to(device)

    # 4. Cargar pesos del estudiante con weights_only=False
    checkpoint_path = os.path.join(args.output_dir, "checkpoint-best-student.pth")
    print(f"Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Preparar los pesos del estudiante para el modelo
    student_weights = checkpoint['model']
    
    # Extraer también el teacher ya que el modelo en modo inferencia TS los requiere a ambos
    checkpoint_teacher = torch.load(os.path.join(args.output_dir, "checkpoint-best.pth"), map_location='cpu', weights_only=False)
    teacher_weights = checkpoint_teacher['model']
    
    # Combinar tal como lo hace main.py en inferencia
    for key in student_weights:
        if 'student' in key:
            teacher_weights[key] = student_weights[key]
            
    model.load_state_dict(teacher_weights, strict=False)
    model.eval()

    # 5. Modo validación igual que test_one_epoch
    predictions = []
    labels = []
    videos = []
    
    print("Evaluando sobre el conjunto de test...")
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Testing final:'
    
    with torch.no_grad():
        for data_iter_step, (samples, grads, targets, label, vid, _) in enumerate(metric_logger.log_every(data_loader_test, args.print_freq, header)):
            videos += list(vid)
            labels += list(label.detach().cpu().numpy())

            samples = samples.to(device)
            grads = grads.to(device)
            targets = targets.to(device)
            
            _, _, _, recon_errors = model(samples, grad_mask=grads, targets=targets, mask_ratio=args.mask_ratio)
            
            # Use strict index 1 for Teacher's reconstruction error
            feature_distance = recon_errors[1].detach().cpu().numpy()
            predictions += list(feature_distance)

    predictions = np.array(predictions)
    labels = np.array(labels)
    videos = np.array(videos)

    print("Calculando AUC y suavizando scores...")
    aucs = []
    filtered_preds = []
    filtered_labels = []
    for vid in np.unique(videos):
        pred = predictions[np.array(videos) == vid]
        
        # Filtro de suavizado temporal gigante original de ShanghaiTech
        pred = filt(pred, range=900, mu=282)

        # Normalización Min-Max por video (Antes del padding para no alterar min/max)
        pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-8)
        pred = np.nan_to_num(pred, nan=0.)

        filtered_preds.append(pred)
        lbl = labels[np.array(videos) == vid]
        filtered_labels.append(lbl)
        
        # Padding original
        lbl = np.array([0] + list(lbl) + [1])
        pred = np.array([0] + list(pred) + [1])

        # Min-len patch (nuestra protección de seguridad extra)
        min_len = min(lbl.shape[0], pred.shape[0])
        lbl = lbl[:min_len]
        pred = pred[:min_len]

        fpr, tpr, _ = metrics.roc_curve(lbl, pred)
        res = metrics.auc(fpr, tpr)
        aucs.append(res)

    macro_auc = np.nanmean(aucs)

    # Micro-AUC (Concatenando todo y aplicando min-len por seguridad general)
    filtered_preds = np.concatenate(filtered_preds)
    filtered_labels = np.concatenate(filtered_labels)

    min_len_micro = min(filtered_labels.shape[0], filtered_preds.shape[0])
    filtered_labels = filtered_labels[:min_len_micro]
    filtered_preds = filtered_preds[:min_len_micro]

    fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
    micro_auc = metrics.auc(fpr, tpr)
    micro_auc = np.nan_to_num(micro_auc, nan=1.0)

    print(f"\n======================================")
    print(f"RESULTADOS FINALES SHANGHAITECH:")
    print(f"Macro-AUC: {macro_auc:.6f}")
    print(f"Micro-AUC: {micro_auc:.6f}")
    print(f"======================================\n")

    # 6. Guardar gráfica
    print("Generando gráfica ROC...")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (Micro-AUC = {micro_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Teacher-Student MAE on ShanghaiTech', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    ruta_salida = args.output_dir
    os.makedirs(ruta_salida, exist_ok=True)
    
    plt.savefig(os.path.join(ruta_salida, 'curva_roc_final.pdf'), format='pdf', dpi=300)
    plt.savefig(os.path.join(ruta_salida, 'curva_roc_final.png'), format='png', dpi=300)
    print(f"Gráficas guardadas en {ruta_salida}/curva_roc_final.png y .pdf")

if __name__ == '__main__':
    main()
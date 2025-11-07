import numpy as np
import matplotlib.pyplot as plt
import copy
from abc import ABC, abstractmethod

# PATRÓN PROTOTYPE: Clase base para clonación de neuronas
class NeuronaPrototype(ABC):
    def __init__(self, input_size=4, output_size=1, learning_rate=0.01, nombre="Neurona Base"):
        self.learning_rate = learning_rate
        self.nombre = nombre
        np.random.seed(42)
        self.pesos = np.random.randn(input_size, output_size) * 0.1
        self.sesgo = np.random.randn(output_size) * 0.1
    
    @abstractmethod
    def clone(self):
        """PATRÓN PROTOTYPE: Método para clonar la neurona"""
        pass
    
    def activacion_sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def mse(self, y_real, y_predicho):
        return np.mean((y_real - y_predicho) ** 2)
    
    def forward(self, entrada):
        suma_ponderada = np.dot(entrada, self.pesos) + self.sesgo
        return self.activacion_sigmoid(suma_ponderada)
    
    def entrenar(self, X, y, epocas=300):
        historial_perdida = []
        for epoca in range(epocas):
            predicciones = self.forward(X)
            perdida = self.mse(y, predicciones)
            historial_perdida.append(perdida)
            
            error = predicciones - y
            gradiente_pesos = np.dot(X.T, error) / len(X)
            gradiente_sesgo = np.mean(error, axis=0)
            
            self.pesos -= self.learning_rate * gradiente_pesos
            self.sesgo -= self.learning_rate * gradiente_sesgo
        
        return historial_perdida

class NeuronaUTEQ(NeuronaPrototype):
    def __init__(self, learning_rate=0.015):
        super().__init__(input_size=4, output_size=1, learning_rate=learning_rate, nombre="Neurona UTEQ")
    
    def clone(self):  # PATRÓN PROTOTYPE: Clonación de neurona UTEQ
        neurona_clonada = NeuronaUTEQ(self.learning_rate)
        neurona_clonada.pesos = copy.deepcopy(self.pesos)
        neurona_clonada.sesgo = copy.deepcopy(self.sesgo)
        return neurona_clonada

class NeuronaMundial(NeuronaPrototype):
    def __init__(self, learning_rate=0.012):
        super().__init__(input_size=4, output_size=1, learning_rate=learning_rate, nombre="Neurona Mundial")
    
    def clone(self):  # PATRÓN PROTOTYPE: Clonación de neurona Mundial
        neurona_clonada = NeuronaMundial(self.learning_rate)
        neurona_clonada.pesos = copy.deepcopy(self.pesos)
        neurona_clonada.sesgo = copy.deepcopy(self.sesgo)
        return neurona_clonada

def generar_datos_uteq(n_samples=600):
    np.random.seed(42)
    print(f"Generando {n_samples} datos de estudiantes UTEQ...")
    
    edad = np.random.normal(21, 2.5, n_samples)
    rendimiento = np.random.normal(7.5, 1.2, n_samples)
    carga_trabajo = np.random.normal(6, 2, n_samples)
    presion_economica = np.random.normal(7, 2.5, n_samples)
    
    edad_norm = np.clip((edad - 18) / 7, 0, 1)
    rendimiento_norm = np.clip(rendimiento / 10, 0, 1)
    carga_norm = np.clip(carga_trabajo / 12, 0, 1)
    presion_norm = np.clip(presion_economica / 10, 0, 1)
    
    X = np.column_stack([edad_norm, rendimiento_norm, carga_norm, presion_norm])
    
    stress_uteq = (0.2 * edad_norm - 0.3 * rendimiento_norm + 0.35 * carga_norm + 
                   0.4 * presion_norm + np.random.normal(0, 0.15, n_samples))
    
    y = np.clip(stress_uteq + 0.2, 0, 1).reshape(-1, 1)
    return X, y

def cargar_datos_mundiales_csv():
    try:
        print("Cargando datos de estudiantes mundiales (Stress.csv)...")
        with open('Stress.csv', 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        datos_procesados = []
        for i, line in enumerate(lines[1:]):
            if i >= 800: break
            try:
                parts = []
                current_part = ""
                in_quotes = False
                
                for char in line:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                parts.append(current_part.strip())
                
                if len(parts) >= 13:
                    edad = 0.3 if '18-22' in parts[0] else 0.7 if '23-26' in parts[0] else 0.5
                    cgpa = 0.3 if '2.50' in parts[5] else 0.5 if '3.00' in parts[5] else 0.7 if '3.40' in parts[5] else 0.4
                    stress_q1 = float(parts[9]) / 4.0 if parts[9].replace('.','').isdigit() else 0.5
                    stress_q2 = float(parts[11]) / 4.0 if parts[11].replace('.','').isdigit() else 0.5
                    stress_value = float(parts[12]) if parts[12].replace('.','').isdigit() else 20
                    stress_norm = min(stress_value / 40.0, 1.0)
                    
                    datos_procesados.append([edad, cgpa, stress_q1, stress_q2, stress_norm])
            except:
                continue
        
        datos = np.array(datos_procesados)
        X, y = datos[:, :4], datos[:, 4:5]
        print(f"Datos mundiales cargados: {len(X)} registros")
        return X, y
        
    except Exception as e:
        print(f"Error al cargar CSV: {e}, usando datos sintéticos...")
        return generar_datos_mundiales_sinteticos(600)

def generar_datos_mundiales_sinteticos(n_samples=600):
    np.random.seed(123)
    edad = np.random.normal(20, 2, n_samples)
    rendimiento = np.random.normal(8, 1, n_samples)
    carga_trabajo = np.random.normal(5, 1.5, n_samples)
    presion_economica = np.random.normal(5, 2, n_samples)
    
    edad_norm = np.clip((edad - 18) / 7, 0, 1)
    rendimiento_norm = np.clip(rendimiento / 10, 0, 1)
    carga_norm = np.clip(carga_trabajo / 10, 0, 1)
    presion_norm = np.clip(presion_economica / 10, 0, 1)
    
    X = np.column_stack([edad_norm, rendimiento_norm, carga_norm, presion_norm])
    
    stress_mundial = (0.25 * edad_norm - 0.25 * rendimiento_norm + 0.3 * carga_norm + 
                      0.25 * presion_norm + np.random.normal(0, 0.12, n_samples))
    
    y = np.clip(stress_mundial, 0, 1).reshape(-1, 1)
    return X, y

def crear_curva_aprendizaje_comparativa(neurona_prototipo, X, y, nombre_dataset):
    n_total = len(X)
    n_train = int(0.8 * n_total)
    
    indices = np.random.permutation(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    tamaños = np.linspace(50, n_train, 8, dtype=int)
    perdidas_entrenamiento, perdidas_validacion = [], []
    
    print(f"\nCreando curva de aprendizaje para {nombre_dataset}...")
    
    for tamaño in tamaños:
        print(f"  Entrenando con {tamaño} muestras...")
        
        # PATRÓN PROTOTYPE: Clonar la neurona para cada experimento
        neurona_experimento = neurona_prototipo.clone()
        
        X_subset = X_train[:tamaño]
        y_subset = y_train[:tamaño]
        
        neurona_experimento.entrenar(X_subset, y_subset, epocas=300)
        
        pred_train = neurona_experimento.forward(X_subset)
        pred_val = neurona_experimento.forward(X_val)
        
        loss_train = neurona_experimento.mse(y_subset, pred_train)
        loss_val = neurona_experimento.mse(y_val, pred_val)
        
        perdidas_entrenamiento.append(loss_train)
        perdidas_validacion.append(loss_val)
    
    return tamaños, perdidas_entrenamiento, perdidas_validacion

def main():
    print("ANÁLISIS COMPARATIVO: ESTRÉS ACADÉMICO UTEQ vs MUNDIAL")
    print("Implementación con Patrón Prototype para Deep Learning")
    
    # PATRÓN PROTOTYPE: Crear prototipos de neuronas especializadas
    prototipo_uteq = NeuronaUTEQ(learning_rate=0.015)
    prototipo_mundial = NeuronaMundial(learning_rate=0.012)
    
    # DATOS DE PRUEBA: ESTUDIANTES UTEQ (Simulados)
    print("\nDATASET 1: Estudiantes UTEQ (Datos de Prueba)")
    X_uteq, y_uteq = generar_datos_uteq(n_samples=600)
    tamaños_uteq, perdidas_train_uteq, perdidas_val_uteq = crear_curva_aprendizaje_comparativa(
        prototipo_uteq, X_uteq, y_uteq, "Estudiantes UTEQ"
    )
    
    # DATOS REALES: ESTUDIANTES MUNDIALES (CSV)
    print("\nDATASET 2: Estudiantes Mundiales (Datos Reales)")
    X_mundial, y_mundial = cargar_datos_mundiales_csv()
    tamaños_mundial, perdidas_train_mundial, perdidas_val_mundial = crear_curva_aprendizaje_comparativa(
        prototipo_mundial, X_mundial, y_mundial, "Estudiantes Mundiales"
    )
    
    # Generar gráficos de curvas de aprendizaje
    print("\nGenerando curvas de aprendizaje comparativas...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico UTEQ
    ax1.plot(tamaños_uteq, perdidas_train_uteq, 'o-', label='Entrenamiento UTEQ', color='blue')
    ax1.plot(tamaños_uteq, perdidas_val_uteq, 's-', label='Validación UTEQ', color='orange')
    ax1.set_xlabel('Muestras de entrenamiento')
    ax1.set_ylabel('Pérdida (MSE)')
    ax1.set_title('Curva de Aprendizaje - Estudiantes UTEQ\n(Datos de Prueba)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    mejora_uteq = ((perdidas_val_uteq[0] - perdidas_val_uteq[-1]) / perdidas_val_uteq[0]) * 100
    ax1.text(0.05, 0.95, f'Mejora: {mejora_uteq:.1f}%\nMuestras: {len(X_uteq)}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
    
    # Gráfico Mundial
    ax2.plot(tamaños_mundial, perdidas_train_mundial, 'o-', label='Entrenamiento Mundial', color='green')
    ax2.plot(tamaños_mundial, perdidas_val_mundial, 's-', label='Validación Mundial', color='red')
    ax2.set_xlabel('Muestras de entrenamiento')
    ax2.set_ylabel('Pérdida (MSE)')
    ax2.set_title('Curva de Aprendizaje - Estudiantes Mundiales\n(Datos Reales - Stress.csv)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    mejora_mundial = ((perdidas_val_mundial[0] - perdidas_val_mundial[-1]) / perdidas_val_mundial[0]) * 100
    ax2.text(0.05, 0.95, f'Mejora: {mejora_mundial:.1f}%\nMuestras: {len(X_mundial)}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('curvas_aprendizaje_uteq_vs_mundial.png', dpi=300, bbox_inches='tight')
    print("Gráficos guardados en 'curvas_aprendizaje_uteq_vs_mundial.png'")
    
    # Resumen final
    print(f"\nRESUMEN COMPARATIVO:")
    print(f"UTEQ - Pérdida final: {perdidas_val_uteq[-1]:.4f}, Estrés promedio: {np.mean(y_uteq):.3f}")
    print(f"Mundial - Pérdida final: {perdidas_val_mundial[-1]:.4f}, Estrés promedio: {np.mean(y_mundial):.3f}")
    
    if np.mean(y_uteq) > np.mean(y_mundial):
        diferencia = (np.mean(y_uteq) - np.mean(y_mundial)) * 100
        print(f"HALLAZGO: Estudiantes UTEQ muestran {diferencia:.1f}% más estrés que mundiales.")
    
    print(f"PATRÓN PROTOTYPE: {len(tamaños_uteq) + len(tamaños_mundial)} clones creados exitosamente.")
    
    plt.show()

if __name__ == "__main__":
    main()
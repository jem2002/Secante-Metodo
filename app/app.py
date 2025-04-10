import customtkinter as ctk
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sympy import pi
import numpy as np

class SecantMethodApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Método de la Secante")
        self.geometry("1200x800")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.create_widgets()
        self.iterations_data = []
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = None

    def create_widgets(self):
        # Frame de entrada
        input_frame = ctk.CTkFrame(self)
        input_frame.pack(pady=10, padx=10, fill="x")

        # Entrada de ecuación
        ctk.CTkLabel(input_frame, text="Función f(x):").grid(row=0, column=0, padx=5, pady=5)
        self.function_entry = ctk.CTkEntry(input_frame, width=300)
        self.function_entry.grid(row=0, column=1, padx=5, pady=5)

        # Entradas para valores iniciales
        ctk.CTkLabel(input_frame, text="x₀:").grid(row=1, column=0, padx=5, pady=5)
        self.x0_entry = ctk.CTkEntry(input_frame)
        self.x0_entry.grid(row=1, column=1, padx=5, pady=5)

        ctk.CTkLabel(input_frame, text="x₁:").grid(row=2, column=0, padx=5, pady=5)
        self.x1_entry = ctk.CTkEntry(input_frame)
        self.x1_entry.grid(row=2, column=1, padx=5, pady=5)

        # Entrada para tasa de error mínima
        ctk.CTkLabel(input_frame, text="Tasa de error mínima:").grid(row=3, column=0, padx=5, pady=5)
        self.tolerance_entry = ctk.CTkEntry(input_frame, placeholder_text="Ej: 0.0001")
        self.tolerance_entry.grid(row=3, column=1, padx=5, pady=5)

        # Botón de cálculo
        self.calculate_button = ctk.CTkButton(input_frame, text="Calcular", command=self.run_secant_method)
        self.calculate_button.grid(row=4, column=0, columnspan=2, pady=10)

        # Tabla de resultados
        columns = ("Iteración", "x_(i-1)", "f(x_(i-1))", "x_i", "f(x_i)", "x_(i+1)", "ε_a")
        self.results_tree = ctk.CTkScrollableFrame(self, width=1100, height=200)
        self.results_tree.pack(pady=10, padx=10)

        # Crear encabezados de tabla
        for col, header in enumerate(columns):
            label = ctk.CTkLabel(self.results_tree, text=header, width=150, corner_radius=0)
            label.grid(row=0, column=col, padx=1, pady=1)

        # Área de gráficos
        self.graph_frame = ctk.CTkFrame(self)
        self.graph_frame.pack(pady=10, padx=10, fill="both", expand=True)

    def run_secant_method(self):
        try:
            # Limpiar resultados anteriores
            self.clear_results()
            
            # Obtener datos de entrada
            f_str = self.function_entry.get()
            x_prev = float(self.x0_entry.get())
            x_curr = float(self.x1_entry.get())

            # Obtener tolerancia
            try:
                tolerance = float(self.tolerance_entry.get())
            except ValueError:
                tolerance = 1e-6
            
            # Definir símbolo y función
            x = sp.symbols('x')
            f_expr = sp.sympify(f_str, locals={'pi': sp.pi})
            f = sp.lambdify(x, f_expr, 'numpy')
            
            # Configurar gráfico
            self.setup_plot(f, f_str, x_prev, x_curr)
            
            # Parámetros de iteración
            max_iter = 50
            self.iterations_data = []
            
            for i in range(max_iter):
                f_prev = f(x_prev)
                f_curr = f(x_curr)
                
                # Calcular siguiente aproximación
                if abs(f_prev - f_curr) < 1e-12:
                    break
                
                x_next = x_curr - (f_curr * (x_prev - x_curr)) / (f_prev - f_curr)
                error = abs((x_next - x_curr) / x_next) if x_next != 0 else float('inf')
                
                # Guardar datos de iteración
                self.iterations_data.append((
                    i+1,
                    x_prev,
                    f_prev,
                    x_curr,
                    f_curr,
                    x_next,
                    error
                ))
                
                # Actualizar tabla
                self.update_table_row(i)
                
                # Actualizar gráfico
                self.update_plot(i, x_prev, x_curr, x_next, f)
                
                # Verificar convergencia
                if error < tolerance:
                    break
                
                # Actualizar valores para siguiente iteración
                x_prev, x_curr = x_curr, x_next
            
            # Mostrar gráfico final
            self.canvas.draw()
            
        except Exception as e:
            self.show_error_message(str(e))

    def setup_plot(self, f, f_str, x0, x1):
        # Limpiar gráfico anterior
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        self.ax.clear()
        
        # Generar datos para la función
        x_vals = np.linspace(min(x0, x1)-2, max(x0, x1)+2, 400)
        y_vals = f(x_vals)
        
        # Graficar función
        self.ax.plot(x_vals, y_vals, label=f'f(x) = {f_str}')
        self.ax.axhline(0, color='gray', linestyle='--')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('f(x)')
        self.ax.legend()
        self.ax.grid(True)
        
        # Crear canvas de matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_plot(self, iter_num, x_prev, x_curr, x_next, f):
        # Graficar puntos y rectas secantes
        self.ax.plot([x_prev, x_curr], [f(x_prev), f(x_curr)], 'r--', alpha=0.5)
        self.ax.plot(x_next, 0, 'go', markersize=8, label=f'Iteración {iter_num+1}')
        self.ax.plot(x_prev, f(x_prev), 'bo', markersize=6)
        self.ax.plot(x_curr, f(x_curr), 'ro', markersize=6)
        
        # Actualizar canvas
        self.canvas.draw()

    def update_table_row(self, row_idx):
        data = self.iterations_data[row_idx]
        for col, value in enumerate(data):
            formatted_value = f"{value:.6f}" if isinstance(value, float) else str(value)
            label = ctk.CTkLabel(self.results_tree, text=formatted_value, width=150, corner_radius=0)
            label.grid(row=row_idx+1, column=col, padx=1, pady=1)

    def clear_results(self):
        for widget in self.results_tree.winfo_children():
            if widget.grid_info()["row"] > 0:
                widget.destroy()
        self.iterations_data = []
        if self.canvas:
            self.ax.clear()
            self.canvas.draw()

    def show_error_message(self, message):
        error_window = ctk.CTkToplevel(self)
        error_window.title("Error")
        ctk.CTkLabel(error_window, text=message).pack(padx=20, pady=20)
        ctk.CTkButton(error_window, text="OK", command=error_window.destroy).pack(pady=10)

if __name__ == "__main__":
    app = SecantMethodApp()
    app.mainloop()
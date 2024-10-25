import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Ground truth
samples = 500
x0 = np.linspace(-1, 1, samples)
x1 = np.linspace(-1, 1, samples)

# Criação da malha
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - x1**2 + x1 - 1

# Adicionando ruído
noise = np.random.normal(0, 0.5, y_truth.shape)
y_noisy = y_truth + noise

# Transformando x0 e x1 em um formato adequado para o regressor
x = np.vstack((x0.ravel(), x1.ravel())).T
y = y_noisy.ravel()

# Dividindo os dados em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Configurando o regressor simbólico
pg = SymbolicRegressor(
    population_size=500,
    generations=20,
    tournament_size=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Treinando o modelo
pg.fit(x_train, y_train)

# Fazendo previsões
y_hat = pg.predict(x_test)

# Avaliando o modelo
print("R² score:", r2_score(y_test, y_hat))
print("Modelo encontrado:\n", pg._program)

# Plotando os resultados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set limits
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-2, 2)

# Plotando a superfície verdadeira
ax.plot_surface(x0, x1, y_truth, rstride=1, cstride=1, color='green', alpha=0.5)


# Criando a grade para as previsões
x0_test, x1_test = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
x_test_grid = np.vstack((x0_test.ravel(), x1_test.ravel())).T
y_pred_grid = pg.predict(x_test_grid).reshape(x0_test.shape)

# Plotando a superfície prevista
ax.plot_surface(x0_test, x1_test, y_pred_grid, color='red', alpha=0.5)

# Calculando a média das previsões
mean_y_hat = np.mean(y_hat)

# Adicionando uma linha para a média das previsões
# Para isso, precisamos escolher um plano na superfície (x0 e x1)
mean_line_x0 = np.linspace(-1, 1, samples)
mean_line_x1 = np.linspace(-1, 1, samples)
mean_line_y = np.full_like(mean_line_x0, mean_y_hat)

# Plotando a linha da média
ax.plot(mean_line_x0, mean_line_x1, mean_line_y, color='yellow', linewidth=3, label='Média das Previsões')

# gráfico
ax.legend()
plt.show()

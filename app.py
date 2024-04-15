# Criado por Alexandre M. Barroso em 2024

import os
import sys
import random
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, ttest_1samp, pearsonr, spearmanr
from scipy.integrate import trapz, nquad, quad, dblquad, simpson
from scipy.optimize import fsolve, minimize, Bounds
from scipy.misc import derivative
from scipy.interpolate import interp1d
from sklearn.feature_selection import mutual_info_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import Symbol, integrate
from tqdm import tqdm
from PIL import Image
import scienceplots
plt.style.use(['science', 'ieee'])
import cProfile
import pstats
import functools
import argparse
profiler = cProfile.Profile()
profiler.enable()

if len(sys.argv) == 1:
    sys.argv.extend([
        '--alvo_F1=421',
        '--a_F1=250',
        '--b_F1=543',
        '--a_F2=908',
        '--b_F2=1987',
        '--limiar_1=600',
        '--limiar_2=345',
        '--L=1',
        '--k_1=1',
        '--k_2=7',
        '--alvo_F1=421',
        '--alvo_F2=1887',
        '--neutro_F1=610',
        '--neutro_F2=1900',
        '--lambda_zero=1.027',
        '--lambda_RP=0.417',
        '--lambda_RA=0.018',
        '--vogais=e',
        '--caminho_do_arquivo=dados.txt'
    ])


def parse_parametros():
    parser = argparse.ArgumentParser()
    parser.add_argument('--otimizar', action='store_true', help="Enable optimization if this flag is set")
    parser.add_argument('--alvo_F1', type=int, default=421)
    parser.add_argument('--alvo_F2', type=int, default=1887)
    parser.add_argument('--limiar_1', type=int, default=600)
    parser.add_argument('--limiar_2', type=int, default=345)
    parser.add_argument('--neutro_F1', type=int, default=610)
    parser.add_argument('--neutro_F2', type=int, default=1900)
    parser.add_argument('--L', type=int, default=1)
    parser.add_argument('--k_1', type=int, default=1)
    parser.add_argument('--k_2', type=int, default=7)
    parser.add_argument('--a_F1', type=int, default=250)
    parser.add_argument('--b_F1', type=int, default=543)
    parser.add_argument('--a_F2', type=int, default=908)
    parser.add_argument('--b_F2', type=int, default=1987)
    parser.add_argument('--lambda_zero', type=float, default=1.027)
    parser.add_argument('--lambda_RA', type=float, default=0.018)
    parser.add_argument('--lambda_RP', type=float, default=0.417)
    parser.add_argument('--vogais', type=str, default="e")
    parser.add_argument('--entrevistados', type=str, default="1,3,5")
    parser.add_argument('--caminho_do_arquivo', type=str, default="dados.txt")
    args = parser.parse_args()
    params_todos = vars(args)

    args.vogais = args.vogais.split(',')
    args.entrevistados = list(map(int, args.entrevistados.split(',')))

    dados_keys = ['alvo_F1', 'alvo_F2', 'limiar_1', 'limiar_2', 'neutro_F1', 'neutro_F2',
                  'L', 'k_1', 'k_2', 'a_F1', 'b_F1', 'a_F2', 'b_F2']

    pesos_keys = ['lambda_zero', 'lambda_RA', 'lambda_RP']

    otimizar = params_todos['otimizar']
    params_arq = {'entrevistados': args.entrevistados, 'vogais': args.vogais, 'caminho_do_arquivo': args.caminho_do_arquivo}
    params_dados = {k: params_todos[k] for k in dados_keys}
    params_pesos = {k: params_todos[k] for k in pesos_keys}

    return params_dados, params_pesos, otimizar, params_arq

def definir_diretorio():
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)
    else:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

definir_diretorio()

def ler_dados(caminho_do_arquivo, vogais, entrevistados):
    dados = pd.read_csv(caminho_do_arquivo, sep=" ", quotechar='"', header=0)
    dados = dados[(dados['F1'] != 'NA') & (dados['F2'] != 'NA')]
    dados['F1'] = pd.to_numeric(dados['F1'])
    dados['F2'] = pd.to_numeric(dados['F2'])
    candidatos = dados[dados["Vogal"].isin(vogais) & dados["Falante"].isin(entrevistados)].copy()
    return candidatos[['F1', 'F2']]

def calcular_kde(dados, largura=0.2):
    print('\nCalculando valores para a f.d.p. dos dados...')
    scaler = StandardScaler()
    valores = np.vstack([dados['F1'], dados['F2']]).T
    valores_normalizados = scaler.fit_transform(valores)
    kde = gaussian_kde(valores_normalizados.T, bw_method=largura)
    limits = [(-np.inf, np.inf), (-np.inf, np.inf)]
    integral, error = nquad(lambda x, y: kde(np.vstack([x, y])), limits)
    print(f"\nChecagem do integral da f.d.p.: {integral}")
    if not np.isclose(integral, 1, atol=1e-3):
        print('   |')
        print("   |--> Valores EDK não estão normalizados.")
    else:
        print('   |')
        print("   |--> Valores EDK estão normalizados.")
    return kde, scaler

def criar_fdp_marginal(kde, params):
    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolucao = params['resolucao']
    grade_F1 = np.linspace(min_F1, max_F1, resolucao)
    grade_F2 = np.linspace(min_F2, max_F2, resolucao)
    F1_mesh, F2_mesh = np.meshgrid(grade_F1, grade_F2)
    pontos_grade = np.vstack([F1_mesh.ravel(), F2_mesh.ravel()])
    valores_kde = kde(pontos_grade).reshape(F1_mesh.shape)
    dx = (max_F1 - min_F1) / (resolucao - 1)
    dy = (max_F2 - min_F2) / (resolucao - 1)
    valores_marginalizados_F1 = np.sum(valores_kde, axis=1) * dy
    integral_marginal_F1 = np.sum(valores_marginalizados_F1) * dx
    if integral_marginal_F1 != 1:
        valores_marginalizados_F1 /= integral_marginal_F1
    funcao_interpolada = interp1d(grade_F1, valores_marginalizados_F1, kind='cubic', fill_value="extrapolate")
    return funcao_interpolada

def arquivo_EDK(kde, params, degrau=0.5):
    limite_F1 = (params['min_F1'], params['max_F1'])
    limite_F2 = (params['min_F2'], params['max_F2'])
    with open('amostra_valores.txt', 'w') as file:
        file.write("F1, F2, Valor EDK\n")
        for F1 in np.arange(limite_F1[0], limite_F1[1], degrau):
            for F2 in np.arange(limite_F2[0], limite_F2[1], degrau):
                valores = [F1, F2]
                valores_kde = kde(valores)[0]
                file.write(f"{F1}, {F2}, {valores_kde}\n")

def integral_fdp(kde, params):
    fator_superior = 1.5
    fator_inferior = 1
    min_F1, max_F1 = params['min_F1'] * fator_inferior, params['max_F1'] * fator_superior
    min_F2, max_F2 = params['min_F2'] * fator_inferior, params['max_F2'] * fator_superior
    def integrando(F1, F2):
        return kde(np.array([F1, F2]))
    integral, _ = nquad(integrando, [[min_F1, max_F1], [min_F2, max_F2]])
    return integral

def verificacao_kde(kde, params, tolerancia=0.02):
    integral = integral_fdp(kde, params)
    return np.isclose(integral, 1.0, atol=tolerancia)

def restricao_articulatoria(F1, F2, kde, params):
    alvo_F1 = params['alvo_F1']
    alvo_F2 = params['alvo_F2']
    neutro_F1 = params['neutro_F1']
    neutro_F2 = params['neutro_F2']
    esforco_alvo = np.sqrt((alvo_F1 - neutro_F1) ** 2 + (alvo_F2 - neutro_F2) ** 2)
    esforco_producao = np.sqrt((F1 - neutro_F1) ** 2 + (F2 - neutro_F2) ** 2)
    distancia = np.sqrt((F1 - alvo_F1) ** 2 + (F2 - alvo_F2) ** 2)
    dif_esforco = (esforco_producao + 1e-6) / (esforco_alvo + 1e-6)
    RA = distancia * dif_esforco
    return RA

def integral_ra(kde, params):
    a_F1 = params['a_F1']
    b_F1 = params['b_F1']
    a_F2 = params['a_F2']
    b_F2 = params['b_F2']
    funcao_ra = lambda F2, F1: np.exp(restricao_articulatoria(F1, F2, kde, params) - kde([F1, F2])[0])
    integral, erro = dblquad(funcao_ra, a_F2, b_F2, lambda F2: a_F1, lambda F2: b_F1)
    return integral

def restricao_perceptual(F1, kde, params):
    limiar_1 = params['limiar_1']
    limiar_2 = params['limiar_2']
    L = params['L']
    k_1 = params['k_1']
    k_2 = params['k_2']
    produto =  (L**2 / ( (1 + np.exp(k_1 * (F1 - limiar_1))) * (1 + np.exp(-k_2 * (F1 - limiar_2))) ) )
    RP = (L - produto)
    return RP

def integral_rp(kde, params, fdp_marginalizada):
    a_F1 = params['a_F1']
    b_F1 = params['b_F1']
    min_F2 = params['a_F2']
    max_F2 = params['b_F2']
    integral, erro = quad(lambda F1: np.exp(restricao_perceptual(F1, kde, params) - fdp_marginalizada(F1)) , a_F1, b_F1)
    return integral

def escore_estabilidade(kde, scaler, fdp_marginalizada, params, lambda_RA, lambda_RP):
    a_F1 = params['a_F1']
    b_F1 = params['b_F1']
    a_F2 = params['a_F2']
    b_F2 = params['b_F2']
    volume = (b_F1 - a_F1) * (b_F2 - a_F2)
    def funcao_harmonia(F1, F2):
        RA = np.exp(restricao_articulatoria(F1, F2, kde, params) - kde([F1, F2])[0])
        RP = np.exp(restricao_perceptual(F1, kde, params) - fdp_marginalizada(F1))
        return lambda_RA * RA + lambda_RP * RP
    def derivada_funcao_harmonia(F1, F2):
        def derivada_F1(F1, F2):
            h = 1e-5
            return (funcao_harmonia(F1 + h, F2) - funcao_harmonia(F1, F2)) / h
        def derivada_F2(F1, F2):
            h = 1e-5
            return (funcao_harmonia(F1, F2 + h) - funcao_harmonia(F1, F2)) / h
        return (derivada_F1(F1, F2) + derivada_F2(F1, F2))
    integral, erro = dblquad(derivada_funcao_harmonia, a_F1, b_F1, lambda F1: a_F2, lambda F1: b_F2, epsabs=1.0e-3, epsrel=1.0e-3)
    estabilidade = integral/volume
    return estabilidade

def entropia_diferencial(kde, params):
    a_F1 = params['min_F1']
    b_F1 = params['max_F1']
    a_F2 = params['min_F2']
    b_F2 = params['max_F2']
    def integrando(F1, F2):
        fdp = kde([F1, F2])[0]
        epsilon = 1e-10
        return -fdp * np.log(fdp + epsilon)
    entropia, erro = nquad(integrando, [[a_F1, b_F1], [a_F2, b_F2]])
    return entropia

def equacao_maxent(F1, F2, fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params):
    RA = np.exp( restricao_articulatoria(F1, F2, kde, params) - kde([F1, F2])[0] )
    RP = np.exp( restricao_perceptual(F1, kde, params) - fdp_marginalizada(F1) )
    formula_maxent = np.exp(-1 - lambda_zero - lambda_RA * RA - lambda_RP * RP)
    return formula_maxent

def calculo_maxent(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params):
    def fdp_maxent(F1, F2):
        maxent = equacao_maxent(F1, F2, fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params)
        return maxent
    return fdp_maxent

def arquivo_maxent(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params):
    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolucao = params['resolucao']
    fdp_maxent = calculo_maxent(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params)
    valores_F1 = np.linspace(min_F1, max_F1, resolucao)
    valores_F2 = np.linspace(min_F2, max_F2, resolucao)
    grade_F1, grade_F2 = np.meshgrid(valores_F1, valores_F2)
    with open('valores_MaxEnt.txt', 'w') as file:
        file.write("F1, F2, Valor MaxEnt\n")
        for i in range(resolucao):
            for j in range(resolucao):
                valor_F1 = grade_F1[i, j]
                valor_F2 = grade_F2[i, j]
                valor_maxent = fdp_maxent(valor_F1, valor_F2)
                file.write(f"{valor_F1}, {valor_F2}, {valor_maxent}\n")

def fdps_otimizacao(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params):
    print('\n---> LAMBDAS <---')
    print('---> Lambda zero:', lambda_zero)
    print('---> Lambda RA:', lambda_RA,)
    print('---> Lambda RP:', lambda_RP)
    print('\n--- INÍCIO DE ITERAÇÃO DA OTIMIZAÇÃO ---')
    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolucao = params['resolucao']
    fdp_maxent = calculo_maxent(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params)
    print('OTIMIZAÇÃO: CÁLCULO DA MAXENT CONCLUÍDO')
    grade_F1 = np.linspace(min_F1, max_F1, resolucao)
    grade_F2 = np.linspace(min_F2, max_F2, resolucao)
    grade_F1, grade_F2 = np.meshgrid(grade_F1, grade_F2)
    print('OTIMIZAÇÃO: CRIAÇÃO DO GRID')
    pontos_grade = np.vstack([grade_F1.ravel(), grade_F2.ravel()])
    print('OTIMIZAÇÃO: STACKING OS PONTOS DAS GRADES EM ARRAY 2D')
    valores_kde = kde(pontos_grade).reshape(grade_F1.shape)
    print('OTIMIZAÇÃO: CRIAÇÃO DOS VALORES DA PDF DOS DADOS CONCLUÍDO')
    valores_maxent = np.zeros_like(grade_F1)
    print('OTIMIZAÇÃO: INICIALIZAR ARRAY 2D PARA VALORES MAXENT CONCLUÍDO')
    print('OTIMIZAÇÃO: INÍCIO DO PREENCHIMENTO DO ARRAY COM VALORES MAXENT')
    for i in range(grade_F1.shape[0]):
        for j in range(grade_F1.shape[1]):
            valor_F1 = grade_F1[i, j]
            valor_F2 = grade_F2[i, j]
            valores_maxent[i, j] = fdp_maxent(valor_F1, valor_F2)
    print('OTIMIZAÇÃO: CRIAÇÃO DOS VALORES MAXENT CONCLUÍDO')
    print('--- FIM DE ITERAÇÃO DA OTIMIZAÇÃO ---')
    return valores_maxent, valores_kde

def criar_fdps(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params):
    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolucao = params['resolucao']
    fdp_maxent = calculo_maxent(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params)
    grade_F1 = np.linspace(min_F1, max_F1, resolucao)
    grade_F2 = np.linspace(min_F2, max_F2, resolucao)
    grade_F1, grade_F2 = np.meshgrid(grade_F1, grade_F2)
    pontos_grade = np.vstack([grade_F1.ravel(), grade_F2.ravel()])
    valores_kde = kde(pontos_grade).reshape(grade_F1.shape)
    valores_maxent = np.zeros_like(grade_F1)
    for i in range(grade_F1.shape[0]):
        for j in range(grade_F1.shape[1]):
            valor_F1 = grade_F1[i, j]
            valor_F2 = grade_F2[i, j]
            valores_maxent[i, j] = fdp_maxent(valor_F1, valor_F2)
    dx = (max_F1 - min_F1) / resolucao
    dy = (max_F2 - min_F2) / resolucao
    integral_maxent = np.sum(valores_maxent) * dx * dy
    print('Integral FDP MaxEnt pré-normalização:', integral_maxent)
    integral_kde = np.sum(valores_kde) * dx * dy
    print('Integral FDP dos dados pré-normalização:', integral_kde)
    if integral_maxent != 1:
        valores_maxent /= integral_maxent
    if integral_kde != 1:
        valores_kde /= integral_kde
    integral_maxent_posn = np.sum(valores_maxent) * dx * dy
    print('Integral FDP MaxEnt pós-normalização:', integral_maxent_posn)
    integral_kde_posn = np.sum(valores_kde) * dx * dy
    print('Integral FDP dos dados pós-normalização:', integral_kde_posn)
    return valores_maxent, valores_kde

def verificacao_fdps(valores_maxent, valores_kde_dimensionados, params):
    fator_superior = 1
    fator_inferior = 1
    min_F1 = params['min_F1'] * fator_inferior
    max_F1 = params['max_F1'] * fator_superior
    min_F2 = params['min_F2'] * fator_inferior
    max_F2 = params['max_F2'] * fator_superior
    resolucao = params['resolucao']
    grade_F1 = np.linspace(min_F1, max_F1, resolucao)
    grade_F2 = np.linspace(min_F2, max_F2, resolucao)
    def calcula_integral_e_verifica_normalizacao(valores):
        integral_F1 = trapz(valores, grade_F1, axis=0)
        integral_total = trapz(integral_F1, grade_F2)
        normalizada = np.isclose(integral_total, 1, atol=0.05)
        return integral_total, normalizada
    integral_maxent, normalizada_maxent = calcula_integral_e_verifica_normalizacao(valores_maxent)
    integral_kde, normalizada_kde = calcula_integral_e_verifica_normalizacao(valores_kde_dimensionados)
    print("\nResultados da Verificação e Comparação das PDFs:")
    print(f"Integral da MaxEnt: {integral_maxent}, Normalizada: {'Sim' if normalizada_maxent else 'Não'}")
    print(f"Integral da EDK: {integral_kde}, Normalizada: {'Sim' if normalizada_kde else 'Não'}")
    return integral_maxent, integral_kde, normalizada_maxent, normalizada_kde

def kullback_leibler(valores_maxent, valores_kde, params):
    epsilon = 1e-10
    valores_maxent = np.clip(valores_maxent, epsilon, None)
    valores_kde = np.clip(valores_kde, epsilon, None)
    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolucao = params['resolucao']
    grade_F1 = np.linspace(min_F1, max_F1, resolucao)
    grade_F2 = np.linspace(min_F2, max_F2, resolucao)
    dx = np.abs(grade_F1[1] - grade_F1[0])
    dy = np.abs(grade_F2[1] - grade_F2[0])
    def normalizar_fdp(valores, dx, dy):
        if valores.ndim == 2:
            soma_integral = np.trapz(np.trapz(valores, dx=dx, axis=1), dx=dy)
        elif valores.ndim == 1:
            soma_integral = np.trapz(valores, dx=dx)
        else:
            raise ValueError("Erro: o input deve ser 1D ou 2D")
        return valores / soma_integral
    valores_maxent_normalizados = normalizar_fdp(valores_maxent, dx, dy)
    valores_kde_normalizados = normalizar_fdp(valores_kde, dx, dy)
    def verificar_integral_fdp_normalizada(valores, dx, dy, nome_fdp):
        if valores.ndim == 2:
            integral = np.sum(np.trapz(np.trapz(valores, dx=dx, axis=1), dx=dy))
        elif valores.ndim == 1:
            integral = np.trapz(valores, dx=dx)
        print(f"Integral da {nome_fdp} normalizada: {integral}")
    verificar_integral_fdp_normalizada(valores_maxent_normalizados, dx, dy, "MaxEnt")
    verificar_integral_fdp_normalizada(valores_kde_normalizados, dx, dy, "EDK")
    p_log_p = valores_kde_normalizados * np.log(valores_kde_normalizados)
    p_log_q = valores_kde_normalizados * np.log(valores_maxent_normalizados)
    def integral_kl(valores, dx, dy):
        integral_intermediaria = np.trapz(valores, dx=dx, axis=1)
        return np.trapz(integral_intermediaria, dx=dy)
    integral_p_log_p = integral_kl(p_log_p, dx, dy)
    integral_p_log_q = integral_kl(p_log_q, dx, dy)
    divergencia_kl = integral_p_log_p - integral_p_log_q
    return divergencia_kl

class OtimizadorKL:
    def __init__(self, kde, params_normalizados_ref):
        self.kde = kde
        self.params = params_normalizados_ref
        self.contador_zero = {'Lambda_zero': 0, 'RA': 0, 'RP': 0}
    def funcao_objetivo(self, lambdas):
        print('Contador zero:', self.contador_zero)
        min_F1 = self.params['min_F1']
        max_F1 = self.params['max_F1']
        min_F2 = self.params['min_F2']
        max_F2 = self.params['max_F2']
        resolucao = self.params['resolucao']
        lambda_zero, peso_RA, peso_RP = lambdas[0], lambdas[1], lambdas[2]
        epsilon = 1e-4
        limiar_zero = 1
        if lambda_zero <= 0:
            self.contador_zero['Lambda_zero'] += 1
            if self.contador_zero['Lambda_zero'] >= limiar_zero:
                lambda_zero = epsilon
        else:
            self.contador_zero['Lambda_zero'] = 0
        if peso_RA <= 0:
            self.contador_zero['RA'] += 1
            if self.contador_zero['RA'] >= limiar_zero:
                peso_RA = epsilon
        else:
            self.contador_zero['RA'] = 0
        if peso_RP <= 0:
            self.contador_zero['RP'] += 1
            if self.contador_zero['RP'] >= limiar_zero:
                peso_RP = epsilon
        else:
            self.contador_zero['RP'] = 0
        valores_maxent, valores_kde = fdps_otimizacao(fdp_marginalizada, lambda_zero, peso_RA, peso_RP, kde, self.params)
        print('\nCalculando divergência KL')
        divergencia_kl = kullback_leibler(valores_maxent, valores_kde, self.params)
        print('\n-----> Divergência KL:', divergencia_kl)
        return divergencia_kl

def extrair_probabilidades(valores_maxent, params, scaler):
    print('\n----')
    min_F1, max_F1 = params['min_F1'], params['max_F1']
    min_F2, max_F2 = params['min_F2'], params['max_F2']
    a_F1, a_F2 = params['a_F1'], params['a_F2']
    b_F1, b_F2 = params['b_F1'], params['b_F2']
    resolucao = params['resolucao']
    grade_F1 = np.linspace(min_F1, max_F1, resolucao)
    grade_F2 = np.linspace(min_F2, max_F2, resolucao)
    dx = grade_F1[1] - grade_F1[0]
    dy = grade_F2[1] - grade_F2[0]
    marginal_F1_fdp = np.trapz(valores_maxent, dx=dy, axis=1)
    marginal_F2_fdp = np.trapz(valores_maxent, dx=dx, axis=0)
    marginal_F1_fdp_normalized = marginal_F1_fdp / np.trapz(marginal_F1_fdp, dx=dx)
    marginal_F2_fdp_normalized = marginal_F2_fdp / np.trapz(marginal_F2_fdp, dx=dy)
    indices_F1 = np.logical_and(grade_F1 >= a_F1, grade_F1 <= b_F1)
    indices_F2 = np.logical_and(grade_F2 >= a_F2, grade_F2 <= b_F2)
    relevant_valores_maxent = valores_maxent[np.ix_(indices_F1, indices_F2)]
    prob_conjunta = np.trapz(np.trapz(relevant_valores_maxent, dx=dx, axis=1), dx=dy)
    prob_F1 = np.trapz(marginal_F1_fdp_normalized[indices_F1], dx=dx)
    prob_F2 = np.trapz(marginal_F2_fdp_normalized[indices_F2], dx=dy)
    print(f"Probabilidade conjunta F1, F2: {prob_conjunta:.9f}")
    print(f"Probabilidade F1: {prob_F1:.12f}")
    print(f"Probabilidade F2: {prob_F2:.12f}")
    print('----')
    return prob_conjunta, prob_F1, prob_F2

params_dados, params_pesos, otimizar, params_arq = parse_parametros()
entrevistados = params_arq['entrevistados']
vogais = params_arq['vogais']
caminho_do_arquivo = params_arq['caminho_do_arquivo']

candidatos = ler_dados(caminho_do_arquivo, vogais, entrevistados)

print('\nArquivo lido com sucesso:', caminho_do_arquivo)

print('\nFalante(s) selecionado(s):', entrevistados)

print('\nVogal(is) selecionada(s):', vogais)


largura_customizada = 0.15
largura_scott = 'scott'
largura_silverman = 'silverman'
kde, scaler = calcular_kde(candidatos)

print('\nOs valores para a f.d.p. foram estimados por EDK com sucesso.')

params_dados['resolucao'] = 1000
params_dados['min_F1'] = min(candidatos['F1'])
params_dados['max_F1'] = max(candidatos['F1'])
params_dados['min_F2'] = min(candidatos['F2'])
params_dados['max_F2'] = max(candidatos['F2'])

params_normalizados = {
    'resolucao':params_dados['resolucao'],
    'alvo_F1': scaler.transform([[params_dados['alvo_F1'], 0]])[0][0],
    'limiar_1': scaler.transform([[params_dados['limiar_1'], 0]])[0][0],
    'limiar_2': scaler.transform([[params_dados['limiar_2'], 0]])[0][0],
    'neutro_F1': scaler.transform([[params_dados['neutro_F1'], 0]])[0][0],
    'neutro_F2': scaler.transform([[params_dados['neutro_F2'], 0]])[0][0],
    'a_F1': scaler.transform([[params_dados['a_F1'], 0]])[0][0],
    'b_F1': scaler.transform([[params_dados['b_F1'], 0]])[0][0],
    'min_F1': scaler.transform([[params_dados['min_F1'], 0]])[0][0],
    'max_F1': scaler.transform([[params_dados['max_F1'], 0]])[0][0],
    'alvo_F2': scaler.transform([[0, params_dados['alvo_F2']]])[0][1],
    'a_F2': scaler.transform([[0, params_dados['a_F2']]])[0][1],
    'b_F2': scaler.transform([[0, params_dados['b_F2']]])[0][1],
    'min_F2': scaler.transform([[0, params_dados['min_F2']]])[0][1],
    'max_F2': scaler.transform([[0, params_dados['max_F2']]])[0][1],
    'L': params_dados['L'],
    'k_1': params_dados['k_1'],
    'k_2': params_dados['k_2'],
}
params_ref = {
        'resolucao':params_dados['resolucao'],
        'alvo_F1': params_dados['alvo_F1'],
        'alvo_F2': params_dados['alvo_F2'],
        'limiar_1': params_dados['limiar_1'],
        'limiar_2': params_dados['limiar_2'],
        'neutro_F1': params_dados['neutro_F1'],
        'neutro_F2': params_dados['neutro_F2'],
        'L': params_dados['L'],
        'k_1': params_dados['k_1'],
        'k_2': params_dados['k_2'],
        'a_F1': candidatos['F1'].min(),
        'b_F1': candidatos['F1'].max(),
        'a_F2': candidatos['F2'].min(),
        'b_F2': candidatos['F2'].max(),
        'min_F1': candidatos['F1'].min(),
        'max_F1': candidatos['F1'].max(),
        'min_F2': candidatos['F2'].min(),
        'max_F2': candidatos['F2'].max(),
    }
params_normalizados_ref = {
    'resolucao':1000,
    'alvo_F1': scaler.transform([[params_ref['alvo_F1'], 0]])[0][0],
    'limiar_1': scaler.transform([[params_ref['limiar_1'], 0]])[0][0],
    'limiar_2': scaler.transform([[params_ref['limiar_2'], 0]])[0][0],
    'neutro_F1': scaler.transform([[params_ref['neutro_F1'], 0]])[0][0],
    'neutro_F2': scaler.transform([[params_ref['neutro_F2'], 0]])[0][0],
    'a_F1': scaler.transform([[params_ref['a_F1'], 0]])[0][0],
    'b_F1': scaler.transform([[params_ref['b_F1'], 0]])[0][0],
    'min_F1': scaler.transform([[params_ref['min_F1'], 0]])[0][0],
    'max_F1': scaler.transform([[params_ref['max_F1'], 0]])[0][0],
    'alvo_F2': scaler.transform([[0, params_ref['alvo_F2']]])[0][1],
    'a_F2': scaler.transform([[0, params_ref['a_F2']]])[0][1],
    'b_F2': scaler.transform([[0, params_ref['b_F2']]])[0][1],
    'min_F2': scaler.transform([[0, params_ref['min_F2']]])[0][1],
    'max_F2': scaler.transform([[0, params_ref['max_F2']]])[0][1],
    'L': params_ref['L'],
    'k_1': params_ref['k_1'],
    'k_2': params_ref['k_2'],
    }

arquivo_EDK(kde, params_normalizados_ref)

print('\nArquivo com amostra de valores normalizados e da f.d.p. foi criado.')

fdp_marginalizada = criar_fdp_marginal(kde, params_normalizados_ref)

print('\nF.d.p. marginalizada foi criada.')

entropia = entropia_diferencial(kde, params_normalizados_ref)

SLSQP = 'SLSQP'
BFGS = 'BFGS'
COBYLA = 'COBYLA'
L_BFGS_B = 'L-BFGS-B'

otimizador = OtimizadorKL(kde, params_normalizados_ref)
lambdas_iniciais = [1, 1, 1]
limites = Bounds([1e-8, 1e-8, 1e-8], [np.inf, np.inf, np.inf])

if otimizar:
    print('\nInicializando otimização...')
    otimizacao = minimize(otimizador.funcao_objetivo, lambdas_iniciais, method='L-BFGS-B', bounds=limites, options={'maxiter': 1000})
    lambdas_otimizados = otimizacao.x
    lambda_zero = lambdas_otimizados[0]
    lambda_RA = lambdas_otimizados[1]
    lambda_RP = lambdas_otimizados[2]
    print('\nOtimização concluída com sucesso.')

else:
    lambda_zero = params_pesos['lambda_zero']
    lambda_RA = params_pesos['lambda_RA']
    lambda_RP = params_pesos['lambda_RP']
    print('\nValores de lambda foram especificados.')

print('\n----')
print('Pesos (lambdas) das restrições:')
print('0. Peso relativo à normalização do MaxEnt:', lambda_zero)
print('1. Peso da restrição perceptual:', lambda_RP)
print('2. Peso da restrição articulatória:', lambda_RA)
print('----')


valor_integral_fdp = integral_fdp(kde, params_normalizados_ref)
violacoes_ra = integral_ra(kde, params_normalizados)
violacoes_ra_pesadas = violacoes_ra*lambda_RA
violacoes_rp = integral_rp(kde, params_normalizados, fdp_marginalizada)
violacoes_rp_pesadas = violacoes_rp*lambda_RP
violacoes_ra_total = integral_ra(kde, params_normalizados_ref)
violacoes_ra_total_pesadas = violacoes_ra_total*lambda_RA
violacoes_rp_total = integral_rp(kde, params_normalizados_ref, fdp_marginalizada)
violacoes_rp_total_pesadas = violacoes_rp_total*lambda_RP
violacoes_ra_normalizado = violacoes_ra_pesadas * 100/violacoes_ra_total_pesadas
violacoes_rp_normalizado = violacoes_rp_pesadas * 100/violacoes_rp_total_pesadas

print('\n----')
print('Candidato:')
print('F1:', params_dados['a_F1'], 'até', params_dados['b_F1'])
print('F2:', params_dados['a_F2'], 'até', params_dados['b_F2'])
print('----')

print('\nCalculando violações...')

print('\nViolações de RP:', violacoes_rp_pesadas)
print('Que representa a parcela das violações totais possíveis (0-100):', violacoes_rp_normalizado)
print('\nViolações de RA:', violacoes_ra_pesadas)
print('Que representa a parcela das violações totais possíveis (0-100):', violacoes_ra_normalizado)

print('\nViolações totais possíveis de RA:', violacoes_ra_total_pesadas)
print('Violações totais possíveis de RP:', violacoes_rp_total_pesadas)

estabilidade = escore_estabilidade(kde, scaler, fdp_marginalizada, params_normalizados, lambda_RA, lambda_RP)
soma_violacoes = violacoes_ra_pesadas + violacoes_rp_pesadas
soma_violacoes_norm = (violacoes_ra_pesadas + violacoes_rp_pesadas) * 100 / (violacoes_ra_total_pesadas + violacoes_rp_total_pesadas)

print('\nCalculando harmonia e estabilidade...')
print('\nEstabilidade:', estabilidade)
print('Escore harmônico:', soma_violacoes)

print('\nCalculando distribuição de probabilidade de máxima entropia...')

f_maxent_normalizado = calculo_maxent(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params_normalizados_ref)
fdp_maxent_formatada, fdp_dados_formatada = criar_fdps(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params_normalizados_ref)
verificacao_fdps(fdp_maxent_formatada, fdp_dados_formatada, params_normalizados_ref)

print('\nDistribuição de probabilidade de máxima entropia criada e normalizada.')

divergencia_kl = kullback_leibler(fdp_maxent_formatada, fdp_dados_formatada, params_normalizados_ref)

print("\nValor da divergência de Kullback-Leibler:", divergencia_kl)

prob_conjunta, prob_F1, prob_F2 = extrair_probabilidades(fdp_maxent_formatada, params_normalizados, scaler)

def gerar_relatorio(caminho_do_arquivo, vogais, entrevistados, largura_customizada, valor_integral_fdp,
                    violacoes_ra, violacoes_rp, violacoes_ra_total, violacoes_rp_total,
                    violacoes_ra_normalizado, violacoes_rp_normalizado, estabilidade, soma_violacoes,
                    entropia, divergencia_kl, prob_conjunta, prob_F1, prob_F2, a_F1, a_F2, b_F1, b_F2,
                    lambda_zero, lambda_RA, lambda_RP):

    with open('relatorio.txt', 'w', encoding='utf-8') as arquivo:
        arquivo.write("Relatório\n")
        arquivo.write("------------------------------------------------\n\n")
        arquivo.write(f"Arquivo utilizado: {caminho_do_arquivo}\n\n")
        arquivo.write(f"Vogal(is): {vogais}\n")
        arquivo.write(f"Falante(s): {entrevistados}\n\n")
        arquivo.write(f'F1: {a_F1} até {b_F1}\n')
        arquivo.write(f'F2: {a_F2} até {b_F2}\n\n')
        arquivo.write(f'Peso da restrição perceptual (RP): {lambda_RP}\n')
        arquivo.write(f'Peso da restrição articulatória (RA): {lambda_RA}\n\n')
        arquivo.write(f"Violações da RP: {violacoes_rp}\n")
        arquivo.write(f"Violações da RA: {violacoes_ra}\n\n")
        arquivo.write(f"Escore harmônico: {soma_violacoes}\n")
        arquivo.write(f"Estabilidade: {estabilidade}\n\n")
        arquivo.write(f"Probabilidade F1: {prob_F1}\n")
        arquivo.write(f"Probabilidade F2: {prob_F2}\n")
        arquivo.write(f"Probabilidade conjunta F1 e F2: {prob_conjunta}\n\n")
        arquivo.write(f"Divergência KL (modelo vs. dados): {divergencia_kl}\n\n")

gerar_relatorio(caminho_do_arquivo, vogais, entrevistados, largura_customizada, valor_integral_fdp,
                violacoes_ra_pesadas, violacoes_rp_pesadas, violacoes_ra_total, violacoes_rp_total,
                violacoes_ra_normalizado, violacoes_rp_normalizado, estabilidade, soma_violacoes,
                entropia, divergencia_kl, prob_conjunta, prob_F1, prob_F2, params_dados['a_F1'], params_dados['a_F2'],
                    params_dados['b_F1'], params_dados['b_F2'], lambda_zero, lambda_RA, lambda_RP)

print('\nArquivo de relatório gerado e análise concluída.')

if __name__ == "__main__":
    args = parse_pesos()
    print(f"Lambda Zero: {args.lambda_zero}")
    print(f"Lambda RA: {args.lambda_RA}")
    print(f"Lambda RP: {args.lambda_RP}")

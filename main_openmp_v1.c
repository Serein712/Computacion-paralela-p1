#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    // Parseo de argumentos (asumiendo ya verificados)
    int r = atoi(argv[1]);           // número de iteraciones
    double omega = atof(argv[2]);     // parámetro omega (0 < omega < 1)
    int N = atoi(argv[3]);            // dimensión de la matriz NxN
    char *nombre_archivo = argv[4];   // archivo binario con la matriz
    
    // LECTURA/ESCRITURA DE FICHEROS - Hilo secuencial
    // ====================================================
    
    // Reserva de memoria para matriz M (NxN)
    double **M = (double**)malloc(N * sizeof(double*));
    for(int i = 0; i < N; i++) {
        M[i] = (double*)malloc(N * sizeof(double));
    }
    
    // leer_matriz(M, N, N, nombre_archivo);
    // O generar aleatoriamente si no existe
    
    // Archivo para guardar normas
    // FILE *archivo_normas = fopen("normas.txt", "w");
    
    
    // MEMORIA PARA EL SISTEMA ITERATIVO
    // ====================================================
    
    // Vectores necesarios (reutilizamos memoria)
    double *x_k = (double*)malloc(N * sizeof(double));       // vector actual
    double *x_k1 = (double*)malloc(N * sizeof(double));      // vector siguiente
    double *Mx = (double*)malloc(N * sizeof(double));        // buffer para M*x_k
    
    // Inicializar x_0 como vector unitario
    for(int i = 0; i < N; i++) {
        x_k[i] = 1.0;
    }
    
    // Calcular y guardar norma de x_0 (secuencial)
    double norma_x0 = 0.0;
    for(int i = 0; i < N; i++) {
        norma_x0 += x_k[i] * x_k[i];
    }
    norma_x0 = sqrt(norma_x0);
    // fprintf(archivo_normas, "||x_0|| = %.17g\n", norma_x0);
    
    
    // MEDICIÓN DE TIEMPO - Inicia antes de la primera región paralela
    // ====================================================
    double tiempo_inicio = omp_get_wtime();
    
    
    // DISEÑO 1: UNA ÚNICA REGIÓN PARALELA para k=0,1,2 (3 primeras iteraciones)
    // ====================================================
    
    int limite_diseno1 = (r >= 3) ? 3 : r;  // Hasta 3 o menos si r < 3
    
    for(int k = 0; k < limite_diseno1; k++) {
        double norma_xk = 0.0;
        
        #pragma omp parallel
        {
            // TÉRMINO 1: Calcular M * x_k
            #pragma omp for
            for(int i = 0; i < N; i++) {
                Mx[i] = 0.0;
                for(int j = 0; j < N; j++) {
                    Mx[i] += M[i][j] * x_k[j];
                }
            }
            
            // TÉRMINO 2: Calcular omega*M*x_k + (1-omega)*x_k
            #pragma omp for
            for(int i = 0; i < N; i++) {
                x_k1[i] = omega * Mx[i] + (1.0 - omega) * x_k[i];
            }
            
            // Calcular norma de x_k para normalización (TÉRMINO 3)
            #pragma omp for reduction(+:norma_xk)
            for(int i = 0; i < N; i++) {
                norma_xk += x_k[i] * x_k[i];
            }
            
            // Solo un hilo calcula la raíz cuadrada
            #pragma omp single
            {
                norma_xk = sqrt(norma_xk);
            }
            
            // TÉRMINO 3: Normalizar dividiendo por ||x_k||
            #pragma omp for
            for(int i = 0; i < N; i++) {
                x_k1[i] = x_k1[i] / norma_xk;
            }
        } // Fin de la única región paralela
        
        // Calcular norma del vector resultante x_{k+1} (secuencial)
        double norma_resultado = 0.0;
        for(int i = 0; i < N; i++) {
            norma_resultado += x_k1[i] * x_k1[i];
        }
        norma_resultado = sqrt(norma_resultado);
        // fprintf(archivo_normas, "||x_%d|| = %.17g\n", k+1, norma_resultado);
        
        // Intercambiar punteros (reutilización de memoria)
        double *temp = x_k;
        x_k = x_k1;
        x_k1 = temp;
    }
    
    
    // DISEÑO 2: TRES REGIONES PARALELAS DISTINTAS para k=3,4,...,r-1
    // ====================================================
    
    for(int k = limite_diseno1; k < r; k++) {
        double norma_xk = 0.0;
        
        // REGIÓN PARALELA 1: Calcular M * x_k
        #pragma omp parallel for
        for(int i = 0; i < N; i++) {
            Mx[i] = 0.0;
            for(int j = 0; j < N; j++) {
                Mx[i] += M[i][j] * x_k[j];
            }
        }
        
        // REGIÓN PARALELA 2: Calcular omega*M*x_k + (1-omega)*x_k Y norma de x_k
        #pragma omp parallel
        {
            #pragma omp for
            for(int i = 0; i < N; i++) {
                x_k1[i] = omega * Mx[i] + (1.0 - omega) * x_k[i];
            }
            
            #pragma omp for reduction(+:norma_xk)
            for(int i = 0; i < N; i++) {
                norma_xk += x_k[i] * x_k[i];
            }
        }
        
        norma_xk = sqrt(norma_xk);
        
        // REGIÓN PARALELA 3: Normalizar dividiendo por ||x_k||
        #pragma omp parallel for
        for(int i = 0; i < N; i++) {
            x_k1[i] = x_k1[i] / norma_xk;
        }
        
        // Calcular norma del vector resultante x_{k+1} (secuencial)
        double norma_resultado = 0.0;
        for(int i = 0; i < N; i++) {
            norma_resultado += x_k1[i] * x_k1[i];
        }
        norma_resultado = sqrt(norma_resultado);
        // fprintf(archivo_normas, "||x_%d|| = %.17g\n", k+1, norma_resultado);
        
        // Intercambiar punteros (reutilización de memoria)
        double *temp = x_k;
        x_k = x_k1;
        x_k1 = temp;
    }
    
    
    // MEDICIÓN DE TIEMPO - Finaliza después de la última región paralela
    // ====================================================
    double tiempo_fin = omp_get_wtime();
    double tiempo_total = tiempo_fin - tiempo_inicio;
    
    printf("Tiempo de ejecución paralela: %.6f segundos\n", tiempo_total);
    
    
    // LIBERACIÓN DE MEMORIA
    // ====================================================
    free(x_k);
    free(x_k1);
    free(Mx);
    
    for(int i = 0; i < N; i++) {
        free(M[i]);
    }
    free(M);
    
    // fclose(archivo_normas);
    
    return 0;
}
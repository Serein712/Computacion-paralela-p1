#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>

void rellenar_matriz(double **matriz, int n_dimension){
	srand(time(NULL));
	for(int i = 0; i < n_dimension ; i++){
		for(int j = 0 ; j < n_dimension ; j++){
			if(i == j) {
				matriz[i][j] = 1.0;
			} else {
				// (double)rand() / RAND_MAX -> 0.00..1 - 1.0
				// .. * 0.02 -> minimo = 0 y maximo = 0.02
				// .. - 0.01 -> minimo = -0.01 y maximo = 0.01
				matriz[i][j] = ( (double)rand() / RAND_MAX ) * 0.02 - 0.01; 
			}
		}
	}
}

void guardar_bin(double **matriz, int n){
	int buffer_tamanyo = n * n;
	double *buffer = (double*)malloc( buffer_tamanyo * sizeof(double));
	if(buffer == NULL){
		printf("Error: guardar_bin -> No se pudo asignar memoria para el buffer\n");
		return;
	}

	int indice = 0;
	for ( int i = 0 ; i < n ; i++){
		memcpy(&buffer[indice], matriz[i], n * sizeof(double));
		indice += n;
	}

	FILE *fichero;
	fichero = fopen("M.bin","wb");
	if (fichero == NULL) {
        printf("Error: guardar_bin -> No se pudo crear/abrir el archivo\n");
        free(buffer);
        return;
    }

	fwrite(buffer, sizeof(double), buffer_tamanyo, fichero);
    fclose(fichero);
    free(buffer);
    printf("\tMatriz guardada correctamente en M.bin\n");
}

void guardar_matriz_txt(double **matriz, int n, const char *nombre_archivo) {
    FILE *archivo = fopen(nombre_archivo, "w");
    if (!archivo) {
        printf("Error: No se pudo crear el archivo %s\n", nombre_archivo);
        return;
    }
    
    // Guardar la matriz en formato texto legible
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(archivo, "%.17g", matriz[i][j]);
            if (j < n - 1) {
                fprintf(archivo, " ");
            }
        }
        fprintf(archivo, "\n");
    }
    
    fclose(archivo);
    printf("Matriz guardada en %s\n", nombre_archivo);
}

int leer_matriz(double **matriz, int n, int m, char *nombre_archivo){
	FILE *archivo = fopen(nombre_archivo,"rb");
	if(!archivo){
		printf("Error: leer_matriz -> no se pudo leer el archivo %s", nombre_archivo);
		return 0;
	}

	int buffer_tamanyo = n * m;
    double *buffer = (double*)malloc(buffer_tamanyo * sizeof(double));
    if (buffer == NULL) {
        printf("Error: leer_matriz -> No se pudo asignar memoria para el buffer\n");
        fclose(archivo);
        return 0;
    }
	
	fread(buffer, sizeof(double), buffer_tamanyo, archivo);
	fclose(archivo);

	int indice = 0;
    for (int i = 0; i < n; i++) {
        memcpy(matriz[i], &buffer[indice], m * sizeof(double));
        indice += m;
    }
	free(buffer);

	return 1;
}

void mostrarMatriz(double **matriz, int filas, int columnas) {
    for(int i = 0; i < filas; i++) {
        for(int j = 0; j < columnas; j++) {
            printf("%8.5f ", matriz[i][j]);
        }
        printf("\n");
    }
}

double calcular_norma_euclidiana(double *vector, int n){
	double norma = 0;
	double suma_cuadrados = 0;
	for(int i = 0 ; i < n ; i++){
		suma_cuadrados += vector[i] * vector[i];
	}
	norma = sqrt(suma_cuadrados);
	return norma;
}

void calcular_producto_MatrizVector(double **M, int n, double *vector, double *resultado) {
	//matriz es cuadrada
	for (int i = 0; i < n; i++) {
		resultado[i] = 0.0;
		for (int j = 0; j < n; j++) {
			resultado[i] += M[i][j] * vector[j];
		}
	}
}

// Función principal x_{k+1} = (ω*M*x_k + (1-ω)*x_k) / ||x_k||
void esquema_iterativo_formula(double **M, int n, double *x_k, double *x_k1, double omega) {
    // Paso 1: Calcular la norma de x_k
    double norma_xk = calcular_norma_euclidiana(x_k, n);
    
    // Verificar que la norma no sea cero para evitar división por cero
    if (norma_xk < 1e-10) {
        printf("Error: norma cercana a cero\n");
        return;
    }
    
    // Paso 2: Calcular M * x_k
    double *Mx = (double*)malloc(n * sizeof(double));
    calcular_producto_MatrizVector(M, n, x_k, Mx);
    
    // Paso 3: Calcular el numerador: ω*M*x_k + (1-ω)*x_k
    for (int i = 0; i < n; i++) {
        x_k1[i] = omega * Mx[i] + (1.0 - omega) * x_k[i];
    }
    
    // Paso 4: Dividir por la norma de x_k (normalización)
    for (int i = 0; i < n; i++) {
        x_k1[i] = x_k1[i] / norma_xk;
    }
    
    free(Mx);
}

int main(int argc, char *argv[]){
	
	if(argc < 4 || argc > 5){
		// argumentos: m ω(0<ω<1) n fichero.bin(opcional)
		printf("Error: Numero de argumentos(%d) incorrecto\n", argc);
		printf("Uso: %s m ω(0<ω<1) n [fichero.bin]\n", argv[0]);
		printf("\t m: numero de iteraciones\n");
		printf("\t ω: parametro de relajacion\n");
		printf("\t n: dimension de matriz\n");
		printf("\t fichero.bin(opcional): fichero con matriz. No introducir ninguno para generar uno con matriz aleatoria\n");
		return 1;
	}

	int m_iteraciones = atoi(argv[1]);
	if(m_iteraciones <= 0){
		printf("Error: m debe ser mayor que 0\n");
		return 1;
	}

	double omega = atof(argv[2]);
	if(omega <= 0 || omega >= 1){
		printf("Error: ω debe cumplir 0<ω<1\n");
		return 1;
	}

	int n_dimension = atoi(argv[3]);
	if(n_dimension <= 0){
		printf("Error: n debe ser mayor que 0\n");
		return 1;
	}

	double **matriz = (double**)malloc(n_dimension * sizeof(double*));
	for(int i = 0 ; i < n_dimension ; i++){
		matriz[i] = (double*)malloc(n_dimension * sizeof(double));
		if(matriz[i] == NULL){
			printf("Error: no se ha podido asignar memoria para la matriz\n");
			for(int j = 0 ; j < i ; j++){
				free(matriz[j]);
			}
			free(matriz);
			return 1;
		}
	}
	
	if(!argv[4]){
		printf("\tNombre de fichero no detectado. Generando la matriz aleatoria..\n");
		rellenar_matriz(matriz, n_dimension);
		guardar_bin(matriz, n_dimension);
		//guardar_matriz_txt(matriz, n_dimension, "M.txt");
	} else {
		char *nombre_archivo = argv[4];
		printf("\tLeyendo el archivo %s..\n", nombre_archivo);
		leer_matriz(matriz, n_dimension, n_dimension, nombre_archivo);
	}
	//mostrarMatriz(matriz, n_dimension, n_dimension);

	//vectores
	double *x_actual = (double*)malloc(n_dimension * sizeof(double));
    double *x_siguiente = (double*)malloc(n_dimension * sizeof(double));
    double *buffer_Mx = (double*)malloc(n_dimension * sizeof(double));
    double *temp;

	// Inicializar x_0 con vector unitario (todos los elementos = 1.0)
	for (int i = 0; i < n_dimension; i++) {
        x_actual[i] = 1.0;
    }

	// Archivo para guardar las normas
    FILE *archivo_normas = fopen("normas.txt", "w");
    
    // Guardar norma de x_0
    double norma = calcular_norma_euclidiana(x_actual, n_dimension);
    fprintf(archivo_normas, "||x_0|| = %.17g\n", norma);
    
    // Realizar m iteraciones
    for (int k = 0; k < m_iteraciones; k++) {
        esquema_iterativo_formula(matriz, n_dimension, x_actual, x_siguiente, omega);
        norma = calcular_norma_euclidiana(x_siguiente, n_dimension);
        fprintf(archivo_normas, "||x_%d|| = %.17g\n", k+1, norma);
        
        // Intercambiar punteros para la siguiente iteración
        temp = x_actual;
        x_actual = x_siguiente;
        x_siguiente = temp;
    }
    
    // Liberar memoria
    free(x_actual);
    free(x_siguiente);
    free(buffer_Mx);
    fclose(archivo_normas);
	for(int i = 0; i < n_dimension; i++) {
        free(matriz[i]);
    }
    free(matriz);
	return 0;
}
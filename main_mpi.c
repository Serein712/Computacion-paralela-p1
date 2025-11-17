#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<mpi.h>

void guardar_bin(double *matriz, int n){
	int buffer_tamanyo = n * n;
	FILE *fichero;
	fichero = fopen("M.bin","wb");
	if (fichero == NULL) {
        printf("Error: guardar_bin -> No se pudo crear/abrir el archivo\n");
        return;
    }
	fwrite(matriz, sizeof(double), buffer_tamanyo, fichero);
    fclose(fichero);
    printf("\tMatriz guardada correctamente en M.bin\n");
}

int leer_matriz(double *matriz, int n, int m, char *nombre_archivo){
	FILE *archivo = fopen(nombre_archivo,"rb");
	if(!archivo){
		printf("Error: leer_matriz -> no se pudo leer el archivo %s", nombre_archivo);
		return 0;
	}
	fread(matriz, sizeof(double), n*m, archivo);
	fclose(archivo);
	return 1;
}

int main(int argc, char *argv[]){
	int nproces, myrank, i;
	int *ElementosCadaProceso, *Desplazamientos;
	int n_dimension, m_iteraciones;
	double omega, uno_menos_omega;
	double *matriz, *matriz_local;
	double *x_actual, *x_siguiente, *temp, *norma, *resultado_parcial;
	char *nombre_archivo;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproces);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	// Proceso 0: ARGUMENTOS
	if(myrank == 0){
		if(argc < 4 || argc > 5){
			// argumentos: m omega(0<omega<1) n fichero.bin(opcional)
			printf("Error: Numero de argumentos(%d) incorrecto\n", argc);
			printf("Uso: %s m omega(0<omega<1) n [fichero.bin]\n", argv[0]);
			printf("\t m: numero de iteraciones\n");
			printf("\t omega: parametro de relajacion\n");
			printf("\t n: dimension de matriz\n");
			printf("\t fichero.bin(opcional): fichero con matriz. No introducir ninguno para generar uno con matriz aleatoria\n");
			return 1;
		}
		m_iteraciones = atoi(argv[1]);
		if(m_iteraciones <= 0){
			printf("Error: m debe ser mayor que 0\n");
			return 1;
		}
		omega = atof(argv[2]);
		if(omega <= 0 || omega >= 1){
			printf("Error: omega debe cumplir 0<omega<1\n");
			return 1;
		}
		n_dimension = atoi(argv[3]);
		if(n_dimension <= 0){
			printf("Error: n debe ser mayor que 0\n");
			return 1;
		}
		matriz = (double*)malloc(n_dimension * n_dimension * sizeof(double));
		if(matriz == NULL){
			printf("Error: no se ha podido asignar memoria para la matriz\n");
			return 1;
		}
		if(!argv[4]){
			printf("\tNombre de fichero no detectado. Generando la matriz aleatoria..\n");
			//rellenar_matriz 
			srand(time(NULL));
			for(int i = 0; i < n_dimension ; i++){
				for(int j = 0 ; j < n_dimension ; j++){
					if(i == j) {
						matriz[i*n_dimension+j] = 1.0;
					} else {
						matriz[i*n_dimension+j] = ( (double)rand() / RAND_MAX ) * 0.02 - 0.01; 
					}
				}
			}
			guardar_bin(matriz, n_dimension); 
		} else {
			nombre_archivo = argv[4];
			printf("\tLeyendo el archivo %s..\n", nombre_archivo);
			leer_matriz(matriz, n_dimension, n_dimension, nombre_archivo);
		}
	}

	// Broadcast de parámetros básicos a todos los procesos
	MPI_Bcast(&n_dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m_iteraciones, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&omega, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// TODOS: CÁLCULO DEL REPARTO 
	ElementosCadaProceso = (int*)malloc(nproces * sizeof(int));
	Desplazamientos = (int*)malloc(nproces * sizeof(int));
	if(ElementosCadaProceso == NULL || Desplazamientos == NULL){
		printf("Error: Proceso %d no pudo asignar memoria\n", myrank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	int filas_base = n_dimension / nproces;
	int resto = n_dimension % nproces;
	int offset = 0;
	for(int i = 0; i < nproces; i++){
		// Los primeros 'resto' procesos reciben 1 fila extra
		ElementosCadaProceso[i] = filas_base + (i < resto ? 1 : 0);
		Desplazamientos[i] = offset;
		offset += ElementosCadaProceso[i];
	}

	// Mostrar reparto de filas
	/*if(myrank == 0){
		printf("=== REPARTO DE FILAS ===\n");
		for(int i = 0; i < nproces; i++){printf("\tProceso %d: %d filas (desde fila %d hasta %d)\n",i, ElementosCadaProceso[i], Desplazamientos[i], Desplazamientos[i] + ElementosCadaProceso[i] - 1);}
	}*/

	// ===== PROCESOS RESERVAR MEMORIA =====
	x_actual = (double*)malloc(n_dimension * sizeof(double));
	x_siguiente = (double*)malloc(n_dimension * sizeof(double));		
	if(!x_actual || !x_siguiente){
		printf("Error: Proceso %d no pudo asignar vectores\n", myrank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if(myrank == 0){
		norma = (double*)malloc((m_iteraciones+1)*sizeof(double));
	}
	

	// Reservar buffer LOCAL para SUS filas
	int filas_locales = ElementosCadaProceso[myrank];

	// ===== DISTRIBUCIÓN DE LA MATRIZ =====
	double tiempo_inicio, tiempo_fin, tiempo_total;
	if (nproces == 1) {  
		tiempo_inicio = MPI_Wtime();
		matriz_local = matriz;   
	}
	else {
		matriz_local = (double*)malloc(filas_locales * n_dimension * sizeof(double));
		if (matriz_local == NULL) {
			printf("Error: Proceso %d no pudo asignar matriz_local\n", myrank);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if(myrank == 0){
			//===== INICIO DE MEDICIÓN DE TIEMPO =====
			tiempo_inicio = MPI_Wtime();

			// Proceso 0: copiar SUS filas
			memcpy(matriz_local,&matriz[Desplazamientos[myrank] * n_dimension],ElementosCadaProceso[myrank] * n_dimension * sizeof(double));
			//printf("Proceso 0: copié mis %d filas\n", ElementosCadaProceso[myrank]);
			for (int p = 1; p < nproces; p++) {
				MPI_Send(&matriz[Desplazamientos[p] * n_dimension],
						ElementosCadaProceso[p] * n_dimension,
						MPI_DOUBLE, p, 10, MPI_COMM_WORLD);
				//printf("Proceso 0: envío completo al proceso %d\n", p);
			}
			
		} else {
			//printf("Proceso %d: esperando recibir %d filas...\n", myrank, filas_locales);
			MPI_Recv(matriz_local, filas_locales * n_dimension, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &status);
			//printf("Proceso %d: recibí mis %d filas correctamente\n", myrank, filas_locales);
		}
	}
	
	for (int i = 0; i < n_dimension; i++) {
		x_actual[i] = 1.0;
	}
	uno_menos_omega = 1.0 - omega;
	resultado_parcial = (double*)malloc(filas_locales * sizeof(double)); // un buffer para parte del resultado
	if(resultado_parcial == NULL){
		printf("Error: Proceso %d no pudo asignar resultado_parcial\n", myrank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	double suma_local_norma, suma_total_norma, norma_k_anterior;
	int inicio = Desplazamientos[myrank];
	int fin = inicio + filas_locales;
	// ===== Iteraciones de 1 a 2 sin comunicaciones colectivas =====
	for(int k = 1; k <= 2; k++){
		// 1. Calcular ||x_{k-1}|| 
		suma_local_norma = 0.0;
		for(int i = inicio; i < fin; i++){ //cada proceso calcula la suma de cuadrados de su rango de filas locales
			suma_local_norma += x_actual[i] * x_actual[i];
		}

		if (myrank == 0) {
			double suma_total = suma_local_norma;
			double suma_recibida;
			for (int p = 1; p < nproces; p++) {
				suma_recibida = 0.0;
				MPI_Recv(&suma_recibida, 1, MPI_DOUBLE, p, 33, MPI_COMM_WORLD, &status);
				suma_total += suma_recibida;
			}
			norma_k_anterior = sqrt(suma_total);
			//printf("Proceso %d: norma_k_anterior = %e\n", myrank, norma_k_anterior);
			norma[k-1] = norma_k_anterior; // Almacenar norma

			for (int p = 1; p < nproces; p++)
				MPI_Send(&norma_k_anterior, 1, MPI_DOUBLE, p, 44, MPI_COMM_WORLD);
		} else {
			MPI_Send(&suma_local_norma, 1, MPI_DOUBLE, 0, 33, MPI_COMM_WORLD);
			MPI_Recv(&norma_k_anterior, 1, MPI_DOUBLE, 0, 44, MPI_COMM_WORLD, &status);
		}
		
		// 2. Calcular x_k localmente
		double inv_norma = 1.0 / norma_k_anterior;
		for (int i = 0; i < filas_locales; i++) {
			double suma = 0.0;
			int base = i * n_dimension;
			for (int j = 0; j < n_dimension; j++)
				suma += matriz_local[base + j] * x_actual[j];
			int fila_global = Desplazamientos[myrank] + i;
			resultado_parcial[i] = (omega * suma + uno_menos_omega * x_actual[fila_global]) * inv_norma;
		}

		// 3. Recolectar x_k (sin colectivas)
		if (myrank == 0) {
			for (int i = 0; i < filas_locales; i++)
				x_siguiente[i] = resultado_parcial[i];
			for (int p = 1; p < nproces; p++) {
				MPI_Recv(&x_siguiente[Desplazamientos[p]],
						ElementosCadaProceso[p], MPI_DOUBLE,
						p, 55, MPI_COMM_WORLD, &status);
			}
			temp = x_actual;
			x_actual = x_siguiente;
			x_siguiente = temp;
		} else {
			MPI_Send(resultado_parcial, filas_locales, MPI_DOUBLE, 0, 55, MPI_COMM_WORLD);
		}

		// 4. Enviar x_k a todos para próxima iteración
		if(nproces > 1) MPI_Bcast(x_actual, n_dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	// ===== Iteraciones de 3 a m con comunicaciones colectivas =====
	for(int k = 3; k <= m_iteraciones; k++){
		// Al inicio: x_actual = x_{k-1}
		//MPI_Bcast(x_actual, n_dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		// 1. Calcular ||x_{k-1}|| 
		suma_local_norma = 0.0;
		for(int i = inicio; i < fin; i++){
			suma_local_norma += x_actual[i] * x_actual[i];
		}
    
		suma_total_norma = 0.0;
		MPI_Allreduce(&suma_local_norma, &suma_total_norma, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		norma_k_anterior = sqrt(suma_total_norma);
		
		// Almacenar ||x_{k-1}|| en norma[k-1]
		if(myrank == 0){
			norma[k-1] = norma_k_anterior;
			//printf("\tNorma de x_%d = %f\n", k-1, norma_k_anterior);
		}
    
		// 2. Calcular x_k = [ω*M*x_{k-1} + (1-ω)*x_{k-1}] / ||x_{k-1}||
		double inv_norma = 1.0 / norma_k_anterior;
		for (int i = 0; i < filas_locales; i++) {
			double suma = 0.0;
			resultado_parcial[i] = 0.0;
			int base = i * n_dimension;
			for (int j = 0; j < n_dimension; j++) {
				suma += matriz_local[base + j] * x_actual[j];
			}
			int fila_global = Desplazamientos[myrank] + i;
			resultado_parcial[i] = (omega * suma + uno_menos_omega * x_actual[fila_global]) * inv_norma;
		}
    

		// 3. Recolectar y compartir x_k completo
    	MPI_Allgatherv(resultado_parcial, filas_locales, MPI_DOUBLE,
                   x_siguiente, ElementosCadaProceso, Desplazamientos,
                   MPI_DOUBLE, MPI_COMM_WORLD);
			
		// 4. Intercambiar punteros
		temp = x_actual;
		x_actual = x_siguiente;  
		x_siguiente = temp;
		
	}
	free(resultado_parcial);

	// Calcular norma del último vector x_m
	if(myrank == 0){
		// ===== FIN DE MEDICIÓN DE TIEMPO =====
		tiempo_fin = MPI_Wtime();
		tiempo_total = tiempo_fin - tiempo_inicio;
		printf("\n=== MÉTRICAS DE RENDIMIENTO ===");
		printf("\nTiempo de ejecución paralelo: %.6f segundos\n", tiempo_total);
		printf("Número de procesos: %d\n", nproces);
		printf("Dimensión de la matriz: %d x %d\n", n_dimension, n_dimension);
		printf("Número de iteraciones: %d\n\n", m_iteraciones);
		
		double suma_final = 0.0;
		for(int i = 0; i < n_dimension; i++){
			suma_final += x_actual[i] * x_actual[i];
		}
		norma[m_iteraciones] = sqrt(suma_final);
		//printf("\tNorma x_%d = %.6e\n", m_iteraciones, norma[m_iteraciones]);

		// Guardar el archivo de normas
		FILE *archivo_normas = fopen("normas.txt", "w");
		for(int i = 0; i <= m_iteraciones; i++){
			fprintf(archivo_normas, "||x_%d|| = %.6e\n", i, norma[i]);
		}
		fclose(archivo_normas);
	}

	// ===== LIBERAR MEMORIA =====
	//printf("Proceso %d: Liberando memoria\n", myrank);
	free(matriz_local); // la 'matriz' se libera con matriz_local si el número de procesos es igual a 1
	free(Desplazamientos);

	if (nproces > 1 && myrank == 0){
		free(matriz);
		
	}
	if(myrank == 0){	
		free(norma);
		printf("=== Ejecución finalizada correctamente ===\n");
	}

	free(ElementosCadaProceso);
	free(x_actual);
	free(x_siguiente);

	MPI_Finalize();
	return 0;
}
#ifndef PATOH_WRAP_H
#define PATOH_WRAP_H

#include "external/patoh/patoh.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

using namespace std;

#define CN 0 // Columns are nets
#define RN 1 // Rows are nets
#define TD 2 // 2D -> totally different hgraph, nets and nnz are the vertices
#define CB 3 // Checkerboard -> CN like, reduces the max communication

#define DEFAULT 0
#define SPEED 1 
#define QUALITY 2

#define CUT 1
#define CON 2

/* #define DEBUG */

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

char filename[2048]={'\0'};  
char partfilename[2048]={'\0'};  
int partvprovided = 0;
int patoh_model = -1;
int patoh_speed = -1;
int patoh_metric = -1;
int patoh_no_parts = -1;
double patoh_imbal = -1;

int cut = -1; 
double imbal = -1;

int intcmp(const void *v1, const void *v2){return (*(int *)v1 - *(int *)v2);}
int rintcmp(const void *v1, const void *v2){return (*(int *)v2 - *(int *)v1);}

char* patoh_models[4] = {(char*) "CN", (char*) "RN", (char*) "2D", (char*) "CB"};
char* patoh_speeds[3] = {(char*) "DEFAULT", (char*) "SPEED", (char*) "QUALITY"};
char* patoh_metrics[3] = {(char*) "XXX", (char*) "CUT", (char*) "CON"};



void wrapPaToH(	int _c, int _n, int *xpins, int *pins, int *cwghts, 
				int *nwghts, int *partvec, int no_cons, int patoh_no_parts) {
	int i, ares;
	PaToH_Parameters args;
	int* partwghts;

	
	int cut_type = -1;
	if(patoh_metric == CUT) {
		cut_type = PATOH_CUTPART;
	} else if(patoh_metric == CON) {
		cut_type = PATOH_CONPART;
	} else {
	  throw "unknown patoh metric value";
	}

	int SBProbType = -1;
	if(patoh_speed == DEFAULT) {
		SBProbType = PATOH_SUGPARAM_DEFAULT;
	} else if(patoh_speed == SPEED) {
		SBProbType = PATOH_SUGPARAM_SPEED;
	} else if (patoh_speed == QUALITY){
		SBProbType = PATOH_SUGPARAM_QUALITY;
	} else {
	  throw "unknown patoh metric value";
	}

	PaToH_Initialize_Parameters(&args, cut_type, SBProbType);


	args._k = patoh_no_parts;
	args.MemMul_Pins += 3;
	args.MemMul_CellNet += 3;
	args.init_imbal = patoh_imbal;
	args.final_imbal = patoh_imbal;

	ares = PaToH_Alloc(&args, _c, _n, no_cons, cwghts, nwghts, xpins, pins);
	if (ares) {
		printf("wrapPaToH: error in allocating memory for PaToH %d\n", ares);
		fflush(stdout);
		exit(1);
	}

	struct timeval tp;
	gettimeofday(&tp, NULL);
	args.seed = tp.tv_sec;
	partwghts = (int*) malloc(sizeof(int) * no_cons * patoh_no_parts);	

#ifdef DEBUG
	printf("Executing PaToH\n");
#endif

	PaToH_Part(&args, _c, _n, no_cons, 0, cwghts, nwghts, xpins, pins, NULL, partvec, partwghts, &cut);

	double avgwght = 0;
	for(i = 0; i < patoh_no_parts; i++) {
		avgwght += partwghts[i];
	}
	avgwght /= patoh_no_parts;

	imbal = 0;
	for(i = 0; i < patoh_no_parts; i++) {
		imbal = max(imbal, partwghts[i]/avgwght);
	}
	imbal -= 1;
	
#ifdef DEBUG
	printf("PaToH cut is %d\t_c=%d _n=%d\tpins=%d\timbal=%.3f\n", cut, _c, _n, xpins[_n], imbal);
#endif

	int cut_count = 0, total_conn = 0, j; 
	int* mark = (int*)malloc(sizeof(int) * patoh_no_parts);
	for(i = 0; i < patoh_no_parts; i++) mark[i] = -1;
	for(i = 0; i < _n; i++) {
	  int con = 0;
	  for(j = xpins[i]; j < xpins[i+1]; j++) {
	    if(mark[partvec[pins[j]]] != i) {
	      mark[partvec[pins[j]]] = i;
	      con++;
	    }
	  }
	  if(con > 1) {total_conn += (con - 1); cut_count++;}
	}	
	free(mark);
	printf("computed conn and cut_count are %d and %d\n", total_conn, cut_count);
		
	
	PaToH_Free();
	free(partwghts);
}

void colNetPart(int* ptrs, int* js, int m, int n, int* partv ) {
	int *degs, *xpins, *pins, *cwghts, *nwghts;
	int  i, p;

	degs = (int *)malloc(sizeof(int) * n);	
	memset(degs, 0, sizeof(int) * n);

	for(i = 0; i < m; i++) {
	  for(p = ptrs[i]; p < ptrs[i+1]; p++) {			
	    degs[js[p]]++;
	  }
	}

	nwghts = (int *) malloc(sizeof(int) * n);	
	for(i = 0; i < n; i++) nwghts[i] = 1;

	xpins = (int *) malloc(sizeof(int) * (n+1));
	xpins[0] = 0;
	for(i = 0; i < n; i++) {
		xpins[i+1] = xpins[i] + degs[i];
		degs[i] = xpins[i];
	}  

	pins = (int*) malloc(sizeof(int) * xpins[n]);	
	for(i = 0; i < m; i++) {
		for(p = ptrs[i]; p < ptrs[i+1]; p++) {
			pins[degs[js[p]]++] = i;
		}
	}

	cwghts = (int *) malloc(sizeof(int) * m);  
	for(i = 0; i < m; i++) {cwghts[i] = ptrs[i+1] - ptrs[i];}

	wrapPaToH(m, n, xpins, pins, cwghts, nwghts, partv, 1, patoh_no_parts);

	/*#ifdef DEBUG
	int noCutNets = 0;
	for(i = 0; i < n; i++) {
	  int pno = partv[pins[xpins[i]]];
	  for(p = xpins[i] + 1; p < xpins[i+1]; p++) {
	    if(partv[pins[p]] != pno) {
	      noCutNets++;
	      break;
	    }
	  }
	}
	printf("no columns in the cut is %d\n", noCutNets);
	#endif*/

	free(xpins);
	free(pins);
	free(nwghts);
	free(cwghts);
	free(degs);
}

void rowNetPart(int* ptrs, int* js, int m, int n, int* partv) {
	int *xpins, *pins, *cwghts, *nwghts;
	int i, p;

	cwghts = (int *) malloc(sizeof(int) * n);
	memset(cwghts,0,sizeof(int) *n);
	for(i = 0; i < m; i++) {
		for(p = ptrs[i]; p < ptrs[i+1]; p++) {			
			cwghts[js[p]]++;
		}
	}

	nwghts = (int *)malloc(sizeof(int) * m);	
	for(i = 0; i < m; i++) nwghts[i] = 1;

	xpins = (int *) malloc(sizeof(int) * (m+1));
	memcpy(xpins, ptrs, sizeof(int) * (m+1));

	pins = (int*) malloc(sizeof(int) * xpins[m]);	
	for(i = 0; i < m; i++) {
		memcpy(pins + xpins[i], js + ptrs[i], sizeof(int) * (ptrs[i+1] - ptrs[i]));
	}

	wrapPaToH(n, m, xpins, pins, cwghts, nwghts, partv, 1, patoh_no_parts);

	free(xpins);
	free(pins);
	free(nwghts);
	free(cwghts);
}

void twoDimPart(int* I, int* J, int m, int n, int nz, int* partv) {
	int *rdegs, *cdegs, *xpins, *pins, *cwghts, *nwghts;
	int  i;
	
	nwghts = (int *) malloc(sizeof(int) * (m+n));	
	for(i = 0; i < (m+n); i++) nwghts[i] = 1;

	cwghts = (int *) malloc(sizeof(int) * nz);  
	for(i = 0; i < nz; i++) cwghts[i] = 1;

	cdegs = (int *)malloc(sizeof(int) * n);	
	memset(cdegs, 0, sizeof(int) * n);  
	rdegs = (int *)malloc(sizeof(int) * m);	
	memset(rdegs, 0, sizeof(int) * m);  
	for(i = 0; i < nz; i++) {
	  rdegs[I[i]]++;
	  cdegs[J[i]]++;
	}

	xpins = (int *) malloc(sizeof(int) * (m+n+1));
	xpins[0] = 0;
	for(i = 0; i < m; i++) {
		xpins[i+1] = xpins[i] + rdegs[i];
	}  
	for(i = m; i < m+n; i++) {
		xpins[i+1] = xpins[i] + cdegs[i-m];
	}

	for(i = 1; i < m-1; i++) rdegs[i] += rdegs[i-1];
	for(i = m-1; i > 0; i--) rdegs[i] = rdegs[i-1]; 
	rdegs[0] = 0;

	for(i = 1; i < n-1; i++) cdegs[i] += cdegs[i-1];
	for(i = n-1; i > 0; i--) cdegs[i] = nz + cdegs[i-1]; 
	cdegs[0] = nz;
	
	pins = (int*) malloc(sizeof(int) * xpins[m+n]);	
	for(i = 0; i < nz; i++) {
		pins[rdegs[I[i]]++] = i;    
		pins[cdegs[J[i]]++] = i;    
	}
 
	wrapPaToH(nz, n+m, xpins, pins, cwghts, nwghts, partv, 1, patoh_no_parts);

	free(xpins);
	free(pins);
	free(nwghts);
	free(cwghts);
	free(rdegs);
	free(cdegs);
}

void chkBrdPart(int* I, int* J, int* ptrs, int* js,
				int m, int n, int* partv) {
	int *degs, *xpins, *pins, *cwghts, *nwghts;
	int *partv_first, *partv_second;
	int  i, p, nz = ptrs[m];
	
	partv_first = (int *)malloc(sizeof(int) * m);	
	partv_second = (int *)malloc(sizeof(int) * n);	
	degs = (int *)malloc(sizeof(int) * max(m,n));	
	nwghts = (int *) malloc(sizeof(int) * max(m,n));	
	xpins = (int *) malloc(sizeof(int) * (max(m,n)+1));
	pins = (int*) malloc(sizeof(int) * nz);	
	
	int k1 = patoh_no_parts, k2 = 1;
	for(i = 1; i <= sqrt(patoh_no_parts); i++) {
		if(patoh_no_parts % i == 0) {
			k1 = i;
			k2 = patoh_no_parts / i;
		}
	}	
	cwghts = (int *) malloc(sizeof(int) * max(m, n*k1));  	
	
	//first partition starts	
	memset(degs, 0, sizeof(int) * n);
	for(i = 0; i < m; i++) {
		for(p = ptrs[i]; p < ptrs[i+1]; p++) {			
			degs[js[p]]++;
		}
	}	
	for(i = 0; i < n; i++) nwghts[i] = 1;	
	xpins[0] = 0;
	for(i = 0; i < n; i++) {
		xpins[i+1] = xpins[i] + degs[i];
		degs[i] = xpins[i];
	}  
	for(i = 0; i < m; i++) {
		for(p = ptrs[i]; p < ptrs[i+1]; p++) {
			pins[degs[js[p]]++] = i;
		}
	}	
	for(i = 0; i < m; i++) {cwghts[i] = ptrs[i+1] - ptrs[i];}
	wrapPaToH(m, n, xpins, pins, cwghts, nwghts, partv_first, 1, k1);

	//first partition ended second starts
	memset(cwghts, 0, sizeof(int) * n * k1);
	for(i = 0; i < m; i++) {
		for(p = ptrs[i]; p < ptrs[i+1]; p++) {						
			cwghts[k1 * js[p] + partv_first[i]]++;
		}
	}		
	
	for(i = 0; i < m; i++) nwghts[i] = 1;	
	memcpy(xpins, ptrs, sizeof(int) * (m+1));	
	memcpy(pins, js, sizeof(int) * nz);	
	wrapPaToH(n, m, xpins, pins, cwghts, nwghts, partv_second, k1, k2);
	
	for(i = 0; i < nz; i++) {
		p = partv_first[I[i]] * k2 + partv_second[J[i]];
		partv[i] = p;
	}
	
	free(xpins);
	free(pins);
	free(nwghts);
	free(cwghts);
	free(degs);
	free(partv_first);
	free(partv_second);
}

#endif

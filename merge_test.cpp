#include <vector>
#include <algorithm>
#include <mpi.h>
#include <iostream>
#include <string>
#include <assert.h>

enum MPI_CONSTS {
  MPI_GLOBAL_N,
  MPI_CSR_ROW_PTR,
  MPI_CSR_COL,
  MPI_CSR_NODES,
  MPI_CUT,
  MPI_PARTITION,
  MPI_CACHE,
  MPI_UPDATED,
  MPI_BP_DIST,
  MPI_BP_SET,
  MPI_LABEL_INDICES,
  MPI_LABELS,
};

using namespace std;

const int n = 10;

const vector<int> cut{0,3,6,9};

struct LabelSet {
	vector<int> vertices;
};

struct PSL {
	vector<LabelSet> labels;
};

void Log(string msg, int pid) {
  cout << "P" << pid << ": " << msg << endl;
}

template <typename T>
void SendData(T *data, int size, int vertex, int to,
                           MPI_Datatype type = MPI_INT32_T) {
  int tag = (vertex << 1);
  int size_tag = tag | 1;

  MPI_Send(&size, 1, MPI_INT32_T, to, size_tag, MPI_COMM_WORLD);

  if (size != 0 && data != nullptr)
    MPI_Send(data, size, type, to, tag, MPI_COMM_WORLD);
}

template <typename T>
void BroadcastData(T *data, int size, int vertex, int pid, int np,
                                MPI_Datatype type = MPI_INT32_T) {
  for (int p = 0; p < np; p++) {
    if (p != pid)
      SendData(data, size, vertex, p, type);
  }
}

template <typename T>
int RecvData(T *&data, int vertex, int from, MPI_Datatype type = MPI_INT32_T) {
  int tag = (vertex << 1);
  int size_tag = tag | 1;
  int size = 0;

  int error_code1, error_code2;

  error_code1 = MPI_Recv(&size, 1, MPI_INT32_T, from, size_tag, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

  if (size != 0) {
    data = new T[size];
    error_code2 = MPI_Recv(data, size, type, from, tag, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
  } else {
    data = nullptr;
  }

  return size;
}

bool MergeCut(vector<vector<int> *> new_labels, PSL &psl, int pid, int np) {

  bool updated = false;
  
  vector<int> compressed_labels;
  vector<int> compressed_label_indices;
  compressed_label_indices.reserve(cut.size()+1);
  compressed_label_indices.push_back(0);
  int last_index = 0;
  for(int i=0; i<cut.size(); i++){
    int u = cut[i];
    auto* new_labels_u = new_labels[u];

    if(new_labels_u != nullptr){
      sort(new_labels_u->begin(), new_labels_u->end());
      compressed_labels.insert(compressed_labels.end(), new_labels_u->begin(), new_labels_u->end());
      last_index += new_labels_u->size();
      compressed_label_indices.push_back(last_index);
    } else {
      compressed_label_indices.push_back(last_index);
    }
  }

  assert(compressed_labels.size() == compressed_label_indices[cut.size()]);
  Log("Compressed labels to size=" + to_string(compressed_labels.size()), pid);

  int * compressed_merged;
  int * compressed_merged_indices;
  int compressed_size = 0;

  if(pid == 0){ // ONLY P0
    vector<int*> all_compressed_labels(np);
    vector<int*> all_compressed_label_indices(np);

    all_compressed_labels[0] = compressed_labels.data();
    all_compressed_label_indices[0] = compressed_label_indices.data();

    for(int p=1; p<np; p++){
      RecvData(all_compressed_label_indices[p], MPI_LABEL_INDICES, p);
      RecvData(all_compressed_labels[p], MPI_LABELS, p);
    }

    vector<vector<int>*> sorted_vecs(cut.size());
#pragma omp parallel default(shared) num_threads(NUM_THREADS) reduction(+ : compressed_size)
    for(int i=0; i<cut.size(); i++){

      vector<int>* sorted_vec = new vector<int>;
      
      for(int p=0; p<np; p++){
        

        int start = all_compressed_label_indices[p][i];
        int end = all_compressed_label_indices[p][i+1];

        if(start == end){
          continue;
        }

        int prev_size = sorted_vec->size();
        sorted_vec->insert(sorted_vec->end(), all_compressed_labels[p] + start, all_compressed_labels[p] + end);

        if(sorted_vec->size() > prev_size){
          inplace_merge(sorted_vec->begin(), sorted_vec->begin() + prev_size, sorted_vec->end());
        }
      }

      auto unique_it = unique(sorted_vec->begin(), sorted_vec->end());
      sorted_vec->erase(unique_it, sorted_vec->end());
      compressed_size += sorted_vec->size(); 
      sorted_vecs[i] = sorted_vec;
    }

    for(int p=1; p<np; p++){
      delete[] all_compressed_labels[p];
      delete[] all_compressed_label_indices[p];
    }

    compressed_merged = new int[compressed_size];
    compressed_merged_indices = new int[cut.size()+1];
    compressed_merged_indices[0] = 0;
    int index = 0;

    for(int i=0; i<cut.size(); i++){
      if(sorted_vecs[i]->size() > 0)
        copy(sorted_vecs[i]->begin(), sorted_vecs[i]->end(), compressed_merged + index);
      
      index += sorted_vecs[i]->size();
      compressed_merged_indices[i+1] = index;
      delete sorted_vecs[i];
    }

    assert(compressed_merged_indices[cut.size()] == compressed_size);

    BroadcastData(compressed_merged_indices, cut.size()+1, MPI_LABEL_INDICES, pid, np);
    BroadcastData(compressed_merged, compressed_size, MPI_LABELS, pid, np);
    Log("Sent merged labels with size=" + to_string(compressed_size), pid);
    
  } else { // ALL EXCEPT P0
    SendData(compressed_label_indices.data(), compressed_label_indices.size(), MPI_LABEL_INDICES, 0);
    SendData(compressed_labels.data(), compressed_labels.size(), MPI_LABELS, 0);
    RecvData(compressed_merged_indices, MPI_LABEL_INDICES, 0);
    compressed_size = RecvData(compressed_merged, MPI_LABELS, 0);
    Log("Recieved merged labels with size=" + to_string(compressed_size), pid);
  }

  if(compressed_size > 0){
    updated = true;

#pragma omp parallel default(shared) num_threads(NUM_THREADS)
    for(int i=0; i<cut.size(); i++){
      int u = cut[i];
      auto& labels_u = psl.labels[u].vertices;
      
      int start = compressed_merged_indices[i];
      int end = compressed_merged_indices[i+1];

      if(start != end){
	Log("Inserting to=" + to_string(u), pid);
        labels_u.insert(labels_u.end(), compressed_merged + start, compressed_merged + end);
      }

    }

  }

  if(pid == 0){
    delete[] compressed_merged;
    delete[] compressed_merged_indices;
  }


  return updated;
}

void PrintLabels(PSL& psl){
	for(int i=0; i<n; i++){
		cout << i << ": ";
		auto label_set = psl.labels[i].vertices;

		for(int label : label_set){
			cout << label << ", ";
		}
		cout << endl;
	}
}

int main(int argc, char** argv){
	MPI_Init(NULL, NULL);

	int pid;
	int np;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	PSL psl;
	psl.labels.resize(n);

	for(int i=0; i<n; i++){
		psl.labels[i].vertices.push_back(i);
		psl.labels[i].vertices.push_back((i+1) % n);
	}

	vector<vector<int>*> new_labels(n, nullptr);

	new_labels[0] = new vector<int>;
	if(pid == 0)
		new_labels[0]->push_back(3);
	else
		new_labels[0]->push_back(6);


	new_labels[6] = new vector<int>;
	if(pid == 0)
		new_labels[6]->push_back(8);




	if(pid == 1){
	  cout << "Before Merge" << endl;
	  PrintLabels(psl);
	}

	MergeCut(new_labels, psl, pid, np);

	if (pid == 1){
	  cout << "After Merge" << endl;
	  PrintLabels(psl);
	}	

	MPI_Finalize();

}

#pragma once
#include <cstddef>
#include <vector>
using namespace std;

#define OFFSET 0;                  // offset for reading data

class Document
{
public:
	Document() { 
		bTrain = NULL;
		trainIx = NULL;
		neighbors = NULL;
		linkGnd = NULL;
		linkTest = NULL;
		linkLossAug = NULL;
		feature = NULL;
	}

	~Document() {
		if ( bTrain != NULL ) delete []bTrain;
		if ( trainIx != NULL ) delete []trainIx;
		if ( neighbors != NULL ) delete []neighbors;
		if ( linkGnd != NULL ) delete []linkGnd;
		if ( linkTest != NULL ) delete []linkTest;
		if ( linkLossAug != NULL ) delete []linkLossAug;
		if ( feature != NULL ) delete []feature;
	}

	// find the index of a document in the neighbor list
	int find_nix(const int &docId) {
		int nIx = -1;
		for ( int i=0; i<num_neighbors; i++ ) {
			if ( neighbors[i] == docId ) {
				nIx = i;
				break;
			}
		}
		return nIx;
	}

public:
	double lhood;
	bool *bTrain;
	int *trainIx; // index in the training list if the link is training.

	int *neighbors;
	int *linkGnd;
	int *linkTest;
	int *linkLossAug;
	int num_neighbors;
	int num_features;
	double *feature;

	int docid;
};


class Corpus
{
public:
	Corpus(void);
public:
	~Corpus(void);

	void read(char* data_filename, const int &nDocs);
	void read2(char* data_filename, const int &nDocs);
	void read2(char* data_filename, char *feature_filename, const int &nDocs, const int &nFeatures);
	void parse_line(char *text, int &docId, 
						vector<int> &neighborIx, vector<int> &neighborLabel);
	void set_train_tag(const int &ratio);
	void set_train_tag(const int &ratio, char *fileName);
	void set_train_tag_rand(const int &ratio, char *fileName);
	void set_train_tag_rand2(const int &ratio, char *fileName);
	void set_train_tag_rand_b(const int &ratio);
	void set_train_tag_rand_sym(const int &ratio, char *fileName);
	
	//Corpus* get_traindata(const int &trainRatio);
	//Corpus* get_testdata(const int &trainRatio);

	void shuffle();

public:
    Document* docs;
    int num_docs;
	int num_links;
	int num_pos_links;
	int num_neg_links;
	int num_train_links; // # of links used for training.

	// Modified by Jiaming
	void get_sample_link(const int r, int &i, int &j);
};

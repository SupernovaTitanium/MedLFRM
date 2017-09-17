// (C) Copyright 2011, Jun Zhu (junzhu [at] cs [dot] cmu [dot] edu)

// This file is part of Logistic.

// Logistic is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// Logistic is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
#include "Corpus.h"
#include "cokus.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <fstream>
#include <vector>
#include <map>
using namespace std;

Corpus::Corpus(void)
{
	docs = NULL;
}

Corpus::~Corpus(void)
{
	if (docs != NULL ) {
		delete[] docs;
	}
}


void Corpus::shuffle()
{
	srand(time(NULL));
	int n = 0;
	for ( n=0; n<num_docs*100; n++ )
	{
		int ix1 = rand() % num_docs;
		int ix2 = rand() % num_docs;
		if ( ix1 == ix2 ) continue;
		
		Document p = docs[ix1];
		docs[ix1] = docs[ix2];
		docs[ix2] = p;
	}
}

//Corpus* Corpus::get_traindata(const int &ratio)
//{
//	Corpus *subc = new Corpus();
//	subc->num_docs = num_docs * 100 / ratio;
//	subc->docs = new Document[subc->num_docs];
//
//	for ( int nd=0; nd<subc->num_docs; nd++ )
//	{
//		subc->docs[nd].num_neighbors = docs[nd].num_neighbors;
//		subc->docs[nd].docid = docs[nd].docid;
//		subc->docs[nd].neighbors = (int*)malloc(sizeof(int) * subc->docs[nd].num_neighbors);
//		subc->docs[nd].linkGnd = (int*)malloc(sizeof(int) * subc->docs[nd].num_neighbors);
//		subc->docs[nd].linkTest = (int*)malloc(sizeof(int) * subc->docs[nd].num_neighbors);
//
//		for ( int i=0; i<docs[nd].num_neighbors; i++ ) {
//			subc->docs[nd].neighbors[i] = docs[nd].neighbors[i];
//			subc->docs[nd].linkGnd[i] = docs[nd].linkGnd[i];
//			subc->docs[nd].linkTest[i] = docs[nd].linkTest[i];
//		}
//	}
//	return subc;
//}
//
//Corpus* Corpus::get_testdata(const int &ratio)
//{
//	Corpus *subc = new Corpus();
//	int ntrain = num_docs * 100 / ratio;;
//	subc->num_docs = num_docs - ntrain;
//	subc->docs = new Document[subc->num_docs];
//
//	for ( int nd=0; nd<subc->num_docs; nd++ )
//	{
//		subc->docs[nd].num_neighbors = docs[ntrain + nd].num_neighbors;
//		subc->docs[nd].docid = docs[ntrain + nd].docid;
//		subc->docs[nd].neighbors = (int*)malloc(sizeof(int) * subc->docs[nd].num_neighbors);
//		subc->docs[nd].linkGnd = (int*)malloc(sizeof(int) * subc->docs[nd].num_neighbors);
//		subc->docs[nd].linkTest = (int*)malloc(sizeof(int) * subc->docs[nd].num_neighbors);
//
//		for ( int i=0; i<docs[ntrain + nd].num_neighbors; i++ ) {
//			subc->docs[nd].neighbors[i] = docs[ntrain + nd].neighbors[i];
//			subc->docs[nd].linkGnd[i] = docs[ntrain + nd].linkGnd[i];
//			subc->docs[nd].linkTest[i] = docs[ntrain + nd].linkTest[i];
//		}
//	}
//	return subc;
//}
void Corpus::set_train_tag(const int &ratio)
{
	int slt_links = num_links * ratio / 100;
	int nIx = 0;
	for ( int i=0; i<num_docs; i++ ) {
		Document *pDoc = &(docs[i]);
		for ( int j=0; j<pDoc->num_neighbors; j++ ) {
			int jIx = pDoc->neighbors[j];

			if ( jIx > i ) { // set the symmetric link structure
				pDoc->bTrain[j] = true;
				for ( int m=0; m<docs[jIx].num_neighbors; m++ ) {
					if ( i == docs[jIx].neighbors[m] ) {
						docs[jIx].bTrain[m] = true;
						break;
					}
				}
				nIx += 2;
			}
			if ( nIx > slt_links ) break;
		}
		if ( nIx > slt_links ) break;
	}
	printf("train-links: %d\n", nIx);
}

void Corpus::set_train_tag_rand(const int &ratio, char *fileName)
{
	srand(time(NULL));
	int uniqueLinks = num_links;
	int slt_links = uniqueLinks * ratio / 100;
	map<int, bool> mpIx;
	while( (int)mpIx.size() < slt_links ) {
		int randIx = randomMT() % uniqueLinks;
		if (mpIx.find( randIx ) == mpIx.end() ) {
			mpIx.insert( make_pair( randIx, true ) );
		}
	}
	//printf("%d %d\n", slt_links, mpIx.size());

	FILE *fptr = fopen(fileName, "w");
	int nIx = 0, trainLinks = 0;
	for ( int i=0; i<num_docs; i++ ) {
		Document *pDoc = &(docs[i]);
		for ( int j=0; j<pDoc->num_neighbors; j++ ) {
			int jIx = pDoc->neighbors[j];

			if ( mpIx.find(nIx) != mpIx.end() ) { // set the symmetric link structure
				pDoc->bTrain[j] = true;
				pDoc->trainIx[j] = trainLinks; // the index of the train data
				trainLinks += 1;
				fprintf(fptr, "%d %d\n", i, jIx);
			}
			nIx ++;
		}
	}
	fclose(fptr);
	num_train_links = trainLinks;
	printf("train-links: %d\n", trainLinks);
}

void Corpus::set_train_tag_rand_sym(const int &ratio, char *fileName)
{
	srand(time(NULL));
	int uniqueLinks = num_docs * (num_docs + 1) / 2;
	int slt_links = uniqueLinks * ratio / 100;
	map<int, bool> mpIx;
	while( mpIx.size() < slt_links ) {
		int randIx = randomMT() % uniqueLinks;
		if (mpIx.find( randIx ) == mpIx.end() ) {
			mpIx.insert( make_pair( randIx, true ) );
		}
	}

	FILE *fptr = fopen(fileName, "w");
	int nIx = 0, trainLinks = 0;
	for ( int i=0; i<num_docs; i++ ) {
		Document *pDoc = &(docs[i]);
		for ( int j=0; j<pDoc->num_neighbors; j++ ) {
			int jIx = pDoc->neighbors[j];

			if ( jIx >= i ) {
				if ( mpIx.find(nIx) != mpIx.end() ) { // set the symmetric link structure
					pDoc->bTrain[j] = true;
					if ( jIx != i ) {
						for ( int m=0; m<docs[jIx].num_neighbors; m++ ) {
							if ( i == docs[jIx].neighbors[m] ) {
								docs[jIx].bTrain[m] = true;
								docs[jIx].trainIx[m] = trainLinks;
								trainLinks += 1;
								fprintf(fptr, "%d %d\n", jIx, i);
								break;
							}
						}
					}
					pDoc->trainIx[j] = trainLinks; // the index of the train data
					trainLinks += 1;
					fprintf(fptr, "%d %d\n", i, jIx);
				}
				nIx ++;
			}
		}
	}
	fclose(fptr);

	num_train_links = trainLinks;
	printf("train-links: %d\n", trainLinks);
}

void Corpus::set_train_tag_rand2(const int &ratio, char *fileName)
{
	srand(time(NULL));

	FILE *fptr = fopen(fileName, "w");
	int /*nIx = 0,*/ trainLinks = 0;
	for ( int i=0; i<num_docs; i++ ) {
		Document *pDoc = &(docs[i]);

		int slt_links = pDoc->num_neighbors * ratio / 100;
		map<int, bool> mpIx;
		while( (int)mpIx.size() < slt_links ) {
			int randIx = rand() % pDoc->num_neighbors;
			if (mpIx.find( randIx ) == mpIx.end() ) {
				mpIx.insert( make_pair( randIx, true ) );
			}
		}

		for ( int j=0; j<pDoc->num_neighbors; j++ ) {
			int jIx = pDoc->neighbors[j];

			if ( mpIx.find(j) != mpIx.end() ) { // set the symmetric link structure
				pDoc->bTrain[j] = true;
				pDoc->trainIx[j] = trainLinks; // the index of the train data
				trainLinks += 1;
				fprintf(fptr, "%d ", jIx);
			}
		}
		fprintf(fptr, "\n");
	}
	fclose(fptr);
	num_train_links = trainLinks;
	printf("train-links: %d\n", trainLinks);
}


void Corpus::set_train_tag(const int &ratio, char *fileName)
{
	FILE *fptr = fopen(fileName, "r");

	int iIx = 0, jIx = 0, trainLinks = 0;
	while (fscanf(fptr, "%d %d", &iIx, &jIx) != EOF ) {
		Document *pDoc = &(docs[iIx]);

		for ( int j=0; j<pDoc->num_neighbors; j++ ) {
			if ( pDoc->neighbors[j] == jIx ) {
				pDoc->bTrain[j] = true;
				pDoc->trainIx[j] = trainLinks; // the index of the train data
				trainLinks += 1;
				break;
			}
		}
	}
	fclose(fptr);

	num_train_links = trainLinks;
	printf("train-links: %d\n", trainLinks);
}

// balanced sampling.
void Corpus::set_train_tag_rand_b(const int &ratio)
{
	srand(time(NULL));
	int uniquePosLinks = num_pos_links;
	int slt_links = uniquePosLinks * ratio / 100;
	map<int, bool> mpIxPos;
	while( (int)mpIxPos.size() < slt_links ) {
		int randIx = rand() % uniquePosLinks;
		if (mpIxPos.find( randIx ) == mpIxPos.end() ) {
			mpIxPos.insert( make_pair( randIx, true ) );
		}
	}
	int uniqueNegLinks = num_neg_links;
	map<int, bool> mpIxNeg;
	while( (int)mpIxNeg.size() < slt_links ) {
		int randIx = rand() % uniqueNegLinks;
		if (mpIxNeg.find( randIx ) == mpIxNeg.end() ) {
			mpIxNeg.insert( make_pair( randIx, true ) );
		}
	}
	//printf("%d %d\n", slt_links, mpIx.size());

	int nIx = 0, trainLinksPos = 0;
	for ( int i=0; i<num_docs; i++ ) {
		Document *pDoc = &(docs[i]);
		for ( int j=0; j<pDoc->num_neighbors; j++ ) {
			int jIx = pDoc->neighbors[j];

			if ( jIx > i && pDoc->linkGnd[j] > 0 ) {
				if ( mpIxPos.find(nIx) != mpIxPos.end() ) { // set the symmetric link structure
					pDoc->bTrain[j] = true;
					for ( int m=0; m<docs[jIx].num_neighbors; m++ ) {
						if ( i == docs[jIx].neighbors[m] ) {
							docs[jIx].bTrain[m] = true;
							break;
						}
					}
					trainLinksPos += 2;
				}
				nIx ++;
			}
		}
	}

	int trainLinksNeg = 0;
	nIx = 0;
	for ( int i=0; i<num_docs; i++ ) {
		Document *pDoc = &(docs[i]);
		for ( int j=0; j<pDoc->num_neighbors; j++ ) {
			int jIx = pDoc->neighbors[j];

			if ( jIx > i && pDoc->linkGnd[j] < 1 ) {
				if ( mpIxNeg.find(nIx) != mpIxNeg.end() ) { // set the symmetric link structure
					pDoc->bTrain[j] = true;
					for ( int m=0; m<docs[jIx].num_neighbors; m++ ) {
						if ( i == docs[jIx].neighbors[m] ) {
							docs[jIx].bTrain[m] = true;
							break;
						}
					}
					trainLinksNeg += 2;
				}
				nIx ++;
			}
		}
	}
	printf("train-links: (%d %d)\n", trainLinksPos, trainLinksNeg);
}

void Corpus::read(char* data_filename, const int &nDocs)
{
	num_docs = nDocs;
	docs = new Document[nDocs];
	
	num_links = 0;
	num_pos_links = 0;
	num_neg_links = 0;

	printf("reading data from %s\n", data_filename);
	int gnd;
	//char buff[4096];
	FILE *fptr = fopen(data_filename, "r");
	for ( int nd=0; nd<nDocs; nd++ ) {
		docs[nd].num_neighbors = nDocs;// - 1;
		docs[nd].docid = nd;

		docs[nd].neighbors = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
		docs[nd].linkGnd   = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
		docs[nd].linkTest  = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
		docs[nd].linkLossAug  = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
		docs[nd].bTrain = (bool*)malloc(sizeof(bool) * docs[nd].num_neighbors);
		docs[nd].trainIx = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
		docs[nd].num_features = 0;
		docs[nd].feature = NULL;
		int ix = 0;
		for ( int i=0; i<nDocs; i++ ) {
			fscanf(fptr, "%d", &gnd);
			//if (i == nd ) continue;     // ignore the self-relationship (need to consider: relation24 in Kinship is self-relation!!)

			docs[nd].neighbors[ix] = i;
			docs[nd].linkGnd[ix] = gnd;
			docs[nd].linkTest[ix] = -1;
			docs[nd].linkLossAug[ix] = -1;
			docs[nd].bTrain[ix] = false;
			docs[nd].trainIx[ix] = -1;

			num_pos_links += docs[nd].linkGnd[ix];
			num_neg_links += (1 - docs[nd].linkGnd[ix]);
			ix ++;
		}
	}
	fclose(fptr);
	num_links = num_pos_links + num_neg_links;
	
	printf("number of docs    : %d\n", num_docs);
	printf("number of links   : %d (%d %d)\n", num_links, num_pos_links, num_neg_links);
}

void Corpus::read2(char* data_filename, const int &nDocs)
{
  num_docs = nDocs;
  docs = new Document[nDocs];

  num_links = 0;
  num_pos_links = 0;
  num_neg_links = 0;

  printf("reading data from %s\n", data_filename);
  int gnd;
  //char buff[4096];
  FILE *fptr = fopen(data_filename, "r");
  for ( int nd=0; nd<nDocs; nd++ ) {
    int num_neighbors;
    fscanf(fptr, "%d", &num_neighbors);
    docs[nd].num_neighbors = num_neighbors;
    docs[nd].docid = nd;

    docs[nd].neighbors = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
    docs[nd].linkGnd   = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
    docs[nd].linkTest  = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
    docs[nd].linkLossAug  = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
    docs[nd].bTrain = (bool*)malloc(sizeof(bool) * docs[nd].num_neighbors);
    docs[nd].trainIx = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
    docs[nd].num_features = 0;
    docs[nd].feature = NULL;

    for ( int i=0; i<num_neighbors; i++ ) {
      int jIx = 0;
      fscanf(fptr, "%d:%d", &jIx, &gnd);

      docs[nd].neighbors[i] = jIx;
      docs[nd].linkGnd[i] = gnd;
      docs[nd].linkTest[i] = -1;
      docs[nd].linkLossAug[i] = -1;
      docs[nd].bTrain[i] = false;
      docs[nd].trainIx[i] = -1;

      num_pos_links += docs[nd].linkGnd[i];
      num_neg_links += (1 - docs[nd].linkGnd[i]);
    }
  }
  fclose(fptr);
  num_links = num_pos_links + num_neg_links;
  printf("number of docs    : %d\n", num_docs);
  printf("number of links   : %d (%d %d)\n", num_links, num_pos_links, num_neg_links);
}


void Corpus::read2(char* data_filename, char *feature_filename, const int &nDocs, const int &nFeatures)
{
	num_docs = nDocs;
	docs = new Document[nDocs];
	
	num_links = 0;
	num_pos_links = 0;
	num_neg_links = 0;

	printf("reading data from %s\n", data_filename);
	int gnd;
	//char buff[4096];
	FILE *fptr = fopen(data_filename, "r");
	for ( int nd=0; nd<nDocs; nd++ ) {
		int num_neighbors;
		fscanf(fptr, "%d", &num_neighbors);
		docs[nd].num_neighbors = num_neighbors;
		docs[nd].docid = nd;

		docs[nd].neighbors = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
		docs[nd].linkGnd   = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
		docs[nd].linkTest  = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
		docs[nd].linkLossAug  = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
		docs[nd].bTrain = (bool*)malloc(sizeof(bool) * docs[nd].num_neighbors);
		docs[nd].trainIx = (int*)malloc(sizeof(int) * docs[nd].num_neighbors);
		docs[nd].num_features = 0;
		docs[nd].feature = NULL;

		for ( int i=0; i<num_neighbors; i++ ) {
			int jIx = 0;
			fscanf(fptr, "%d:%d", &jIx, &gnd);

			docs[nd].neighbors[i] = jIx - 1;
			docs[nd].linkGnd[i] = gnd;
			docs[nd].linkTest[i] = -1;
			docs[nd].linkLossAug[i] = -1;
			docs[nd].bTrain[i] = false;
			docs[nd].trainIx[i] = -1;

			num_pos_links += docs[nd].linkGnd[i];
			num_neg_links += (1 - docs[nd].linkGnd[i]);
		}
	}
	fclose(fptr);
	num_links = num_pos_links + num_neg_links;

	fptr = fopen(feature_filename, "r");
	for ( int nd=0; nd<nDocs; nd++ ) {
		docs[nd].num_features = nFeatures;
		docs[nd].feature = (double*)malloc(sizeof(double) * nFeatures);

		for ( int i=0; i<nFeatures; i++ ) {
			double fval = 0;
			fscanf(fptr, "%lf", &fval);
			docs[nd].feature[i] = fval;
		}
	}
	fclose(fptr);
	
	printf("number of docs    : %d\n", num_docs);
	printf("number of links   : %d (%d %d)\n", num_links, num_pos_links, num_neg_links);
}

void Corpus::parse_line(char *text, int &docId, 
						vector<int> &neighborIx, vector<int> &neighborLabel)
{
	string str(text);

	// segment to sub-strings.
	vector<string> subStr;
	int nprepos = 0;
	int npos = str.find(" ");
	while (npos != (int)str.npos ) {
		subStr.push_back( str.substr(nprepos, npos-nprepos) );
		nprepos = npos + 1;
		npos = str.find(" ", nprepos);
	}
	subStr.push_back( str.substr( nprepos ) );

	// parse the ground-truth label
	docId = atoi(subStr[0].c_str());

	// parse features
	neighborIx.clear();
	neighborLabel.clear();
	for ( int i=1; i<(int)subStr.size(); i++ ) {
		str = subStr[i];
		int nSepPos = str.find(":");
		
		if ( nSepPos != (int)str.npos ) {
			int nIx = atoi( str.substr(0, nSepPos).c_str() );
			int label = atoi( str.substr(nSepPos+1).c_str() );

			neighborIx.push_back( nIx );			
			neighborLabel.push_back( label );
		}
	}
}


// Jiaming: This function serves to get the corresponding sample link of index r,
//		to retrieve the Document indices of the link.
// This is a rather slow O(n) method. We can get to O(log n), but I am currently too lazy
//		to care about that.
void Corpus::get_sample_link(const int k_r, int &iIdx, int &jIdx) {
	int r = k_r;
	for (int i = 0; i < num_docs; i++) {
		if (r > docs[i].num_neighbors) {
			r -= docs[i].num_neighbors;
		} else {
			iIdx = i;
			jIdx = r;
			return;
		}
	}
	return;
}

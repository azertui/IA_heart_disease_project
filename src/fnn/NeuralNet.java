package fnn;

import java.util.Arrays;

/**
 * 1-hidden neural network for _classification_
 *
 */

public class NeuralNet {
  private float[][] trainingData;
  private float[][] testingData;
  private int nbInstances;
  private int nbFeatures;
  private int nbClasses;
  private int batchSize;
  
  private float[][] X_train, Y_train, X_test, Y_test, W1, W2, b1, b2;
  private float eta = 0.01f;

  /**
   * Class constructor
   * @param data      data read from input file
   * @param batchSize number of instances in each batch for training
   * @param K         number of output class (e.g. Iris dataset => 3)
   */
  public NeuralNet(float[][] data, int batchSize, int K /*nb classes*/) {
    nbInstances = data.length;
    nbFeatures  = data[0].length;
    this.batchSize = batchSize;
    this.nbClasses = K;
    
    int trainingSize = (int) (nbInstances*0.75);
    int testingSize  = nbInstances-trainingSize;
    trainingData = new float[trainingSize][nbFeatures];
    testingData  = new float[testingSize][nbFeatures];
    
    // Copy data into training & testing set
    for (int i = 0; i < trainingSize; i++)
      for (int j = 0; j < data[0].length; j++)
        trainingData[i][j] = data[i][j];
    
    for (int i = 0; i < testingSize; i++)
      for (int j = 0; j < data[0].length; j++)
        testingData[i][j] = data[i+trainingSize][j];
    
    
    X_train = new float[this.nbFeatures-1][this.batchSize];  // -1 because the last column is the label
    Y_train = new float[this.nbClasses][this.batchSize];
    
    X_test  = new float[this.nbFeatures-1][this.batchSize];
    Y_test  = new float[this.nbClasses][this.batchSize];
    
    W1 = new float[5][13]; NNLib.initMatrix(W1);
    b1 = new float[5][1]; for (int i = 0; i < b1.length; i++) Arrays.fill(b1[i], 0.f);
    
    W2 = new float[1][5]; NNLib.initMatrix(W2);
    b2 = new float[1][1]; for (int i = 0; i < b2.length; i++) Arrays.fill(b2[i], 0.f);
  }
  
  @SuppressWarnings("unused")
  private void printData(float[][] data){
    for (int i = 0; i < data.length;i++){
      for (int j = 0; j < data[0].length; j++)
        System.out.print(data[i][j] + " ");
      System.out.println();
    }
  }
  
  
  /**
   * Load one instance of data
   * Separate features and labels #dataIndex in X and Y
   * @param dataSet   training or testing
   * @param X         matrix for inputs
   * @param Y         matrix for outputs
   * @param dataIndex index in dataSet of data to copy
   */
  private void loadAttributesAndLabels(float[][] dataSet, float[][] X, float[][] Y, int dataIndex, int batchSize){
    // 1. Load data attributes in X
    int k = 0; // index to load the features (data[ex][i] => feature #i of instance #ex)
    int indexInBatch = dataIndex%batchSize;
    /* TODO Part 3, Q.2 */
    for(k=0;k<dataSet[indexInBatch].length-1;k++) {
    	X[k][indexInBatch]=dataSet[dataIndex][k];
    }
    // 2. Load data labels in Y (create a one-hot for class prediction)
    //Load label in Y
    Y[0][indexInBatch]=dataSet[dataIndex][k];
  }
  
  /**
   * Shuffle the set of training data
   * Fisherâ€“Yates shuffle :
   * https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
   * Note : NNLib.rnd.nextInt(max) returns an integer in [0;max[
   */
  private void shuffleTrainingData() {
    /* TODO Part 4, Q.1 */
	  /*
	   * for i from nâˆ’1 downto 1 do
     j â†� random integer such that 0 â‰¤ j â‰¤ i
     exchange a[j] and a[i]
	   */
	  int j;float[] t;
	  for(int i=0;i<trainingData.length;i++) {
		  j=(int) Math.random()*i;
		  t=new float[trainingData.length];
		  t=trainingData[j];
		  trainingData[j]=trainingData[i];
		  trainingData[i]=t;
	  }
  }
  
  /**
   * Feed the testing set to the model and measure error / accuracy
   * @returns cost on testing set
   */
  public float testPrediction() {
    //int currentDataIndex = 0;
    float cost = 0.f;
    int countPredictions = 0,VP = 0,FP = 0,FN = 0,VN = 0;
    
    /* TODO Part 4, Q.3 */
    for(int i=0;i<testingData.length;i++)
    	loadAttributesAndLabels(testingData,X_test,Y_test,i,batchSize);
    //On effectue une passe
    float[][] Z1,A1,Z2,A2;
    Z1 = NNLib.addVec(NNLib.mult(W1, X_test), b1);
	A1=NNLib.tanh(Z1);
	Z2=NNLib.addVec(NNLib.mult(W2, A1), b2);
	A2=NNLib.softmax(Z2);
	//Calcul
	int batchSize = A2[0].length;
	int K = Y_test.length; // # classes
    for (int k = 0; k < K; k++)
      for (int c = 0; c < batchSize; c++) {
        cost += Y_test[k][c]*Math.log(A2[k][c]);
        if(NNLib.checkPrediction(A2, Y_test, c)) {
        	countPredictions++;
        	if(Y_test[k][c] == 1.0 )
        		VP ++;
        	else
        		VN++;
        }
        else if(Y_test[k][c] == 1.0 )
        	FN++;
        else
        	FP++;
      }
    cost = -(1.f/batchSize)*cost;
    System.out.println("predict="+(countPredictions));
    System.out.println("Pourcentage de prédictions correctes ="+(countPredictions/batchSize)*100);
    System.out.println("Pourcentage de faux positifs ="+(FP/batchSize)*100);
    System.out.println("Pourcentage de faux négatifs ="+(FN/batchSize)*100);
    System.out.println("Pourcentage de vrais positifs ="+(VP/batchSize)*100);
    System.out.println("Pourcentage de vrais négatifs ="+(VN/batchSize)*100);

    
    float accuracy = 100.f*countPredictions/(K*batchSize);
    System.out.println("  CE cost on test data: "  + cost);
    System.out.println("  Accuracy on test data: " + accuracy);
    return accuracy;
  }
  
  /**
   * Perform 1 epoch of training
   *  1/ load a minibatch of data into X_train and Y_train
   *  2/ forward pass
   *  3/ compute the gradient of the loss
   *  4/ update the parameters
   */
  private float trainingEpoch(){
    int seenTrainingData = 0;
    @SuppressWarnings("unused")
    float trainingError  = 0.f;
    //float testingError   = 0.f;
    
    shuffleTrainingData(); // shuffle the data before training
    float [][] Z1,A1,Z2,A2,delta1,delta2,dW2,dW1,db1,db2;
    /* TODO Part 4, Q.2 */
    for(;seenTrainingData<trainingData.length;seenTrainingData++) {
    	for(int i=0;i<trainingData.length;i++) {
    		loadAttributesAndLabels(trainingData, this.X_train, this.Y_train,i ,this.batchSize);
    	}
    	//propagation avant
    	Z1=NNLib.addVec(NNLib.mult(W1, X_train), b1);
    	A1=NNLib.tanh(Z1);
    	Z2=NNLib.addVec(NNLib.mult(W2, A1), b2);
    	A2=NNLib.softmax(Z2);
    	//calcul de l'erreur
    	trainingError=NNLib.crossEntropy(Y_train,A2);
    	//System.out.println("training error: "+trainingError);
    	//Retropropagation de l'erreur
    	delta2=NNLib.subtract(A2,trainingError);
    	//delta2=NNLib.subtract(trainingError,A2);
    	dW2=NNLib.mult(delta2,NNLib.transpose(A1));
    	db2=delta2;
    	delta1=NNLib.hadamard(NNLib.mult(W2, delta2),NNLib.tanhDeriv(A1));
    	if(seenTrainingData==0) {
    		//System.out.println("print de début pour vérif:");
    		//System.out.println(testPrediction());
    	}
    	dW1=NNLib.mult(delta1, NNLib.transpose(X_train));
    	db1=delta1;
    	//Mise a jour des paramÃ¨tres
    	W2=NNLib.subtract(W2, NNLib.mult(dW2, this.eta));
    	b2=NNLib.subtract(b2, NNLib.mult(db2, eta));
    	W1=NNLib.subtract(W1, NNLib.mult(dW1, this.eta));
    	b1=NNLib.subtract(b1, NNLib.mult(db1, eta));
    }
    return testPrediction();
  }
  
  /**
   * Train the neural network
   * @param nbEpoch number of epoch to run
   */
  public void train(int nbEpoch){
    String trainingProgress = "";
    for (int e = 0; e < nbEpoch; e++){
      System.out.println(" [ Epoch " +e+ "]");
      trainingProgress += e + "," + trainingEpoch() + "\n";
    }
    // Export the error per epoch in output file
    DataLib.exportDataToCSV("training.out", trainingProgress);
  }

}

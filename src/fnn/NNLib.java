package fnn;

import java.util.Random;

public class NNLib {
  public static Random rnd = new Random(42);
  
  public static void initMatrix(float[][] A){
    for (int i = 0; i < A.length; i++)
      for (int j = 0; j < A[0].length; j++)
        A[i][j] = rnd.nextFloat(); // [0;1]
  }
  
  public static void printMatrix(float[][] A){
    System.out.println("Size: m=" + A.length + " ;n=" + A[0].length);
    for (int i = 0; i < A.length; i++){
      for (int j = 0; j < A[0].length; j++)
        System.out.print(A[i][j] + " ");
      System.out.println();
    }
  }
  
  public static void printMatrix(float[][] A, String message){
    System.out.println("["+message+"] Size: m=" + A.length + " ;n=" + A[0].length);
    for (int i = 0; i < A.length; i++){
      for (int j = 0; j < A[0].length; j++)
        System.out.print(A[i][j] + " ");
      System.out.println();
    }
  }
  
  public static void checkMatrix(float[][] A, String message){
    for (int i = 0; i < A.length; i++)
      for (int j = 0; j < A[0].length; j++)
        if (Float.isNaN(A[i][j]))
          throw new ArithmeticException("A["+i+"]["+j+"] is Nan ("+message+")");
  }
  
  /* ************************************* *
   * ACTIVATION FUNCTIONS
   * ************************************* */
  
  /**
   * Compute the tanh of weighted inputs
   * @param Z matrix of weighted inputs
   * @return matrix of activations
   */
  public static float[][] tanh(float[][] Z){
    float[][] activations = new float[Z.length][Z[0].length];
    for (int i = 0; i < Z.length; i++)
      for (int j = 0; j < Z[0].length; j++)
        activations[i][j] = (float) (2.f/(1+Math.exp(-2.f*Z[i][j])) - 1);
    return activations;
  }
  
  /**
   * Normalisation des attributs num�riques
   * @param data	donn�es � traiter
   * @param indices	tableau contenant les indices des colonnes � normaliser
   */
  public static void normalize(float[][] data, int[] indices) {
	  float moyenne=0.f, e_type=0.f;
	  for(int i=0; i<indices.length;i++) {
		  //calcul de la moyenne
		  for(int k=0;k<data[0].length;k++) {
			  moyenne+=data[indices[i]][k];
		  }
		  moyenne=moyenne/data[0].length;
		  //calcul de l'�cart-type
		  for(int k=0;k<data[0].length;k++) {
			  e_type+=Math.pow((data[indices[i]][k]-moyenne), 2);
		  }
		  e_type=e_type/data[0].length;
		  e_type=(float) Math.sqrt(e_type);
		  //Normalisation
		  for(int k=0;k<data[0].length;k++) {
			  data[indices[i]][k]=(data[indices[i]][k]-moyenne)/e_type;
		  }
	  }
  }
  /**
   * Compute the derivative of tanh function
   * 
   *       /!\ assumes tanh has already been applied
   *       
   * @param tanhA float matrix of tanh activations
   * @return matrix
   */
  public static float[][] tanhDeriv(float[][] tanhA) {
    return subtract(1, power(tanhA, 2));
  }
 
  
  /**
   * Compute the hyperbolic secant
   * @param float z
   * @return float sech(z)
   */
  public static float sech(float z) {
	  return (float) ( 2.f / (Math.exp(z)+Math.exp(-z)));
  }
  
  /**
   * Activate weighted input with ReLU function
   * @param Z matrix of activated weighted input
   * @return matrix of activations
   */
  public static float[][] relu(float[][] Z){
    float[][] activations = new float[Z.length][Z[0].length];
    for (int i = 0; i < Z.length; i++)
      for (int j = 0; j < Z[0].length; j++)
        activations[i][j] = Z[i][j] > 0 ? Z[i][j] : 0;
    return activations;
  }
  
  /**
   * Activate weighted input with Binary step function
   * @param Z matrix of activated weighted input
   * @return matrix of activations
   */
  public static float[][] bstep(float[][] Z){
    float[][] activations = new float[Z.length][Z[0].length];
    for (int i = 0; i < Z.length; i++)
      for (int j = 0; j < Z[0].length; j++)
        activations[i][j] = Z[i][j] >= 0 ? 1 : 0;
    return activations;
  }
  
  /**
   * Derivative of ReLU function
   * @param A matrix of activations
   * @return matrix
   */
  public static float[][] reluDeriv(float[][] A){
    float[][] C = new float[A.length][A[0].length];
    for (int i = 0; i < A.length; i++)
      for (int j = 0; j < A[0].length; j++)
        C[i][j] = A[i][j] > 0 ? 1 : 0;
    return C;
  }
  
  /**
   * Apply a softmax on weighted inputs (for output only)
   * @param Z matrix of weighted inputs
   * @return matrix of activations
   */
  public static float[][] softmax(float[][] Z){
    float[][] softA = new float[Z.length][Z[0].length];
    for (int i = 0; i < softA[0].length; i++){ // for each instance in current batch
      for (int k = 0; k < softA.length; k++){ // for each class k
        float sum = 0;
        for (int c = 0; c < softA.length; c++) sum += Math.exp(Z[c][i]); // sum all class c
        softA[k][i] = (float) (Math.exp(Z[k][i])/sum); // and compute soft for class k
      }
    }
    return softA;
  }
  
  /**
   * Apply a sigmoid on weighted inputs (for output only)
   * @param Z matrix of weighted inputs
   * @return matrix of activations
   */
  public static float[][] sigmoid(float[][] Z){
    float[][] sigA = new float[Z.length][Z[0].length];
    for(int i=0;i<sigA.length;i++) {
    	for(int j=0;j<sigA[0].length;j++) {
    		sigA[i][j]= 1.0f/(1.0f + (float)Math.exp(-Z[i][j]));
    	}
    }
    return sigA;
  }
  
  /**
   * Test if hypothesis matches actual labels
   * @param Yhat hypothesis
   * @param Y    actual labels
   * @param indexInBatch pick instance in batch
   * @return
   */
  public static boolean checkPrediction(float[][] Yhat, float[][] Y, int indexInBatch){
    if(Yhat[0][indexInBatch]>=0.5f && Y[0][indexInBatch]==1.0f) {
    	return true;
    }else if(Yhat[0][indexInBatch]<0.5f && Y[0][indexInBatch]==0.0f) {
    	return true;
    }else{
    	return false;
    }
  }
  
  /* ************************************* *
   * LOSS FUNCTION
   * ************************************* */
  /**
   * Cross entropy loss (use when output is softmax)
   * The loss increases when the predicted probability diverges from the true label
   * @param yHat hypothesis
   * @param y    actual labels
   * @return error
   */
  public static float crossEntropy(float[][] yHat, float[][] y){ 
    int K = y.length; // # classes
    float cost = 0.f;
    int batchSize = yHat[0].length;
    for (int k = 0; k < K; k++)
      for (int c = 0; c < batchSize; c++)
        cost += y[k][c]*Math.log(yHat[k][c]);
    return -(1.f/batchSize)*cost;
  }
  
  /* ************************************* *
   * OPERATIONS ON MATRICES
   * ************************************* */
  public static String getMatDim(float[][] A){
    return A.length +"x" + A[0].length;
  }
  
  /**
   * Add 2 matrices
   * @param A
   * @param B
   * @return C sum of A and B
   */
  public static float[][] add(float[][] A, float[][] B){
    float[][] C = new float[A.length][A[0].length];
    for (int i = 0; i < A.length; i++){
      for (int j = 0; j < A[0].length; j++){
        C[i][j] = A[i][j] + B[i][j];
      }
    }
    return C;
  }
  
  /**
   * Add matrix A and vector B
   * @param A
   * @param B vector stored as a matrix (has 1 column only)
   * @return C
   */
  public static float[][] addVec(float[][] A, float[][] B){
    float[][] C = new float[A.length][A[0].length];
    for (int i = 0; i < A.length; i++){
      for (int j = 0; j < A[0].length; j++){
        C[i][j] = A[i][j] + B[i][0];
      }
    }
    return C;
  }
  
  /**
   * returns axis i of A as a matrix stored vector
   * @param A
   * @param i
   * @return V
   */
  public static float[][] getAxis(float[][] A, int i){
    float[][] C = new float[A.length][1];
    for (int k = 0; k < A.length; k++){
        C[k][0] = A[k][i];
    }
    return C;
  }
  
  /**
   * Add matrix A with a scalar
   * @param A
   * @param scalar
   * @return C
   */
  public static float[][] add(float[][] A, float scalar){
    float[][] C = new float[A.length][A[0].length];
    for (int i = 0; i < A.length; i++){
      for (int j = 0; j < A[0].length; j++){
        C[i][j] = A[i][j] + scalar;
      }
    }
    return C;
  } 
  
  /**
   * Subtract matrix B from matrix A
   * @param A
   * @param B
   * @return C
   */
  public static float[][] subtract(float[][] A, float[][] B){
    float[][] C = new float[A.length][A[0].length];
    for (int i = 0; i < A.length; i++){
      for (int j = 0; j < A[0].length; j++){
        C[i][j] = A[i][j] - B[i][j];
      }
    }
    return C;
  }
  
  /**
   * Subtract a scalar from matrix A
   * @param A
   * @param scalar
   * @return C
   */
  public static float[][] subtract(float[][] A, float scalar){
    float[][] C = new float[A.length][A[0].length];
    for (int i = 0; i < A.length; i++){
      for (int j = 0; j < A[0].length; j++){
        C[i][j] = A[i][j] - scalar;
      }
    }
    return C;
  }
  
  /**
   * Subtract matrix A from a scalar
   * @param scalar
   * @param A
   * @return
   */
  public static float[][] subtract(float scalar, float[][] A){
    float[][] C = new float[A.length][A[0].length];
    for (int i = 0; i < A.length; i++){
      for (int j = 0; j < A[0].length; j++){
        C[i][j] = scalar - A[i][j];
      }
    }
    return C;
  } 
  
  /**
   * Compute a component wise multiplication of matrices A and B
   * @param A
   * @param B
   * @return C
   */
  public static float[][] hadamard(float[][] A, float[][] B){
    float[][] C = new float[A.length][A[0].length];
    for (int i = 0; i < A.length; i++){
      for (int j = 0; j < A[0].length; j++){
        C[i][j] = A[i][j] * B[i][j];
      }
    }
    return C;
  }
  
  /**
   * Multiply matrices A and B
   * @param A m*n
   * @param B n*p
   * @return C m*p
   * @throws RuntimeException Dimensions don't match for matrix mult.
   */
  public static float[][] mult(float[][] A, float[][] B) throws RuntimeException{
    int m = A.length;
    int n = A[0].length;
    if (B.length != n) throw new RuntimeException("Invalid dimensions -- A: " + getMatDim(A) + " B:" + getMatDim(B));
    int p = B[0].length;
    float[][] C = new float[m][p];
    for (int i = 0; i < m; i++)
      for (int j = 0; j < p; j++)
          for (int k = 0; k < n; k++)
              C[i][j] += A[i][k] * B[k][j];
    return C;
  }
  
  /**
   * Multiply matrix A with a scalar
   * @param A
   * @param scalar
   * @return C
   */
  public static float[][] mult(float[][] A, float scalar){
    float[][] C = new float[A.length][A[0].length];
    for (int i = 0; i < A.length; i++)
      for (int j = 0; j < A[0].length; j++)
        C[i][j] = A[i][j] * scalar;
    return C;
  }
  
  /**
   * Compute the power p of matrix A
   * @param A
   * @param p
   * @return C
   */
  public static float[][] power(float[][] A, int p){
    float[][] C = new float[A.length][A[0].length];
    for (int i = 0; i < A.length; i++)
      for (int j = 0; j < A[0].length; j++)
        C[i][j] = (float) Math.pow(A[i][j],p);
    return C;
  }
  
  /**
   * Transpose matrix A
   * @param A  m*n
   * @return C n*n
   */
  public static float[][] transpose(float[][] A){
    int m = A.length;
    int n = A[0].length;
    float[][] C = new float[n][m];
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        C[j][i] = A[i][j];
    return C;
  }

}

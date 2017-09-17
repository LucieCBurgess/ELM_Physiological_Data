package dev.elm

import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix, SingularValueDecomposition, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRowMatrix, RowMatrix}
import breeze.linalg.pinv
import breeze.linalg.{sum, DenseMatrix => BDM}
import org.apache.spark.sql.DataFrame

/**
  * Created by lucieburgess on 26/08/2017.
  * Attribution to: https://stackoverflow.com/questions/29869567/spark-distributed-matrix-multiply-and-pseudo-inverse-calculating
  */
class MatrixInverse {

  def computeInverse(X: RowMatrix): DenseMatrix = {
    val nCoef = X.numCols.toInt
    val svd = X.computeSVD(nCoef, computeU = true)
    if (svd.s.size < nCoef) {
      sys.error(s"RowMatrix.computeInverse called on singular matrix.")
    }

    // Create the inv diagonal matrix from S
    val invS = DenseMatrix.diag(new DenseVector(svd.s.toArray.map(x => math.pow(x,-1))))

    // U cannot be a RowMatrix
    val U = new DenseMatrix(svd.U.numRows().toInt,svd.U.numCols().toInt,svd.U.rows.collect.flatMap(x => x.toArray))

    // Or use this version - to BlockMatrix not working
    //val U = svd.U.toBlockMatrix.toLocaMatrix().multiply(DenseMatrix.eye(svd.U.numRows().toInt)).transpose

    // If you could make V distributed, then this may be better. However its alreadly local...so maybe this is fine.
    val V = svd.V
    // inv(X) = V*inv(S)*transpose(U)  --- the U is already transposed.
    V.multiply(invS).multiply(U)
  }

  /** https://stackoverflow.com/questions/29869567/spark-distributed-matrix-multiply-and-pseudo-inverse-calculating
    * M^+ = (M^T.M)^{-1}.M^T where M^T is the transpose of M. Assumes M is of full column rank
    * @param M the matrix to be inverted
    * @return the Moore-Penrose inverse of M
    */
  def computePseudoInverse(M: DenseMatrix) :DenseMatrix = {
      ???
  }

//  def convertSparkDMtoBreezeDM(X: DataFrame) :BDM() = {
//    val arr: Array = X.rows.map(x => x.toArray).collect.flatten }

}

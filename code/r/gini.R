## compute the gini index, for validation evaluation
gini = function(y_true, y_pred){
  if(length(y_true) != length(y_pred)){stop('size must match')}
  n_samples = length(y_true)
  arr = data.frame(ytrue = y_true, ypred = y_pred)
  true_order = arr[order(arr$ytrue, decreasing = TRUE),]$ytrue
  pred_order = arr[order(arr$ypred, decreasing = TRUE),]$ytrue
  L_true = cumsum(true_order) / sum(true_order)
  L_pred = cumsum(pred_order) / sum(pred_order)
  L_ones = seq(0,1,len = n_samples)
  G_true = sum(L_ones - L_true)
  G_pred = sum(L_ones - L_pred)
  return(G_pred/G_true)
}

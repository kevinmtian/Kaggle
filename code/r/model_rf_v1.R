# This script creates a sample submission using a Random Forest
# and also plots the feature importance from the trained model.
#
# Click "fork" to run this script yourself and make tweaks
install.packages('readr')


setwd("/Users/mutian/Desktop/CodeStation/KaggleProp/code/r")
library(ggplot2)
library(randomForest)
library(readr)
set.seed(1)

cat("Reading data\n")
train <- read_csv("../../data/train/train.csv")
test <- read_csv("../../data/test/test.csv")

# We'll convert all the characters to factors so we can train a randomForest model on them
extractFeatures <- function(data) {
  character_cols <- names(Filter(function(x) x=="character", sapply(data, class)))
  for (col in character_cols) {
    data[,col] <- as.factor(data[,col])
  }
  return(data)
}

trainFea <- extractFeatures(train)
testFea  <- extractFeatures(test)

cat("Training model\n")
rf <- randomForest(trainFea[,3:34], trainFea$Hazard, ntree=100, imp=TRUE, sampsize=10000, do.trace=TRUE)
cat("Making predictions\n")
submission <- data.frame(Id=test$Id)
submission$Hazard <- predict(rf, extractFeatures(testFea[,2:33]))
write_csv(submission, "../../data/pred/1_random_forest_benchmark.csv")

cat("Plotting variable importance\n")
imp <- importance(rf, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

p <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="#53cfff") +
  coord_flip() + 
  theme_light(base_size=20) +
  xlab("Importance") +
  ylab("") + 
  ggtitle("Random Forest Feature Importance\n") +
  theme(plot.title=element_text(size=18))

ggsave("../../data/eda/2_feature_importance.png", p, height=12, width=8, units="in")
rm(list = ls(all=TRUE))
#setwd('.....')

## Agenda 
#Read in the data
#Data Pre-processing
#Build Multiple Models
#Stack 'em up
#Report Metrics of the various Models on Test Data

cancer_data <- read.csv("C:/Users/srinivas/Music/Insofe/Mod1_and_Mod2/14.10/20171014_Batch31_CSE7305c_Ensemble_LabActivity/cancer_diagnosis.csv",header = TRUE, sep = ',')
str(cancer_data)
head(cancer_data)
tail(cancer_data)

#Let's convert the 'Cancer' column into a factor, because it was read in as a numeric attribute (1 is if the patient has cancer 
#and 0 is if the patient does not have cancer)
cancer_data$Cancer <- as.factor(cancer_data$Cancer)
#remove irrelevant attribute 'id' from the dataset
cancer_data <- cancer_data[ , !(colnames(cancer_data) %in% "id")]
#check for missing values
sum(is.na(cancer_data))

#Split the dataset into train and test using using stratified sampling using the caret package

library(caret)
set.seed(1234)
index_train <- createDataPartition(cancer_data$Cancer, p = 0.7, list = F)
pre_train <- cancer_data[index_train, ]
pre_test <- cancer_data[-index_train, ]

index_val <- createDataPartition(pre_train$Cancer, p = 0.7, list = F)
pre_train_2 <- pre_train[index_val, ]
pre_val <- pre_train[-index_val, ]

#use preProcess() from caret to standardize variables using data points in training data
std_method <- preProcess(pre_train_2, method = c("center", "scale"))
train_data <- predict(std_method, pre_train_2)
test_data <- predict(std_method, pre_test)
val_data <- predict(std_method, pre_val)

#Build Mutiple models: Linear SVM, kNN, Decision Trees
#SVM
library(e1071)
model_svm <- svm(Cancer ~ . , train_data, kernel = "linear")
summary(model_svm)

preds_svm <- predict(model_svm, val_data)
confusionMatrix(preds_svm, val_data$Cancer)
preds_train_svm <- model_svm$fitted

#kNN
model_knn <- knn3(Cancer ~ . , train_data, k = 5)
preds_k <- predict(model_knn, val_data)
preds_knn <- ifelse(preds_k[, 1] > preds_k[, 2], 0, 1)
confusionMatrix(preds_knn, val_data$Cancer)

preds_train_k <- predict(model_knn, train_data)
preds_train_knn <- ifelse(preds_train_k[, 1] > preds_train_k[, 2], 0, 1)

#Decision Trees using CART
library(rpart)
model_dt <- rpart(Cancer ~ . , train_data)
preds_dt <- predict(model_dt, val_data)
preds_tree <- ifelse(preds_dt[, 1] > preds_dt[, 2], 0, 1)
confusionMatrix(preds_tree, val_data$Cancer)

preds_train_dt <- predict(model_dt)
preds_train_tree <- ifelse(preds_train_dt[, 1] > preds_train_dt[, 2], 0, 1)

#Building a stacked ensemble
#collate all the predictions on the train and validation datasets into a dataframe

train_preds_df <- data.frame(svm = preds_train_svm, knn = preds_train_knn,
                             tree = preds_train_tree, Cancer = train_data$Cancer)
val_preds_df <- data.frame(svm = preds_svm, knn = preds_knn,
                           tree = preds_tree, Cancer = val_data$Cancer)

#combine those two dataframes together and convert target variable into factor
stack_df <- rbind(train_preds_df, val_preds_df)
stack_df$Cancer <- as.factor(stack_df$Cancer)

#Use the sapply() function to convert all the variables other than the target variable into a numeric type
numeric_st_df <- sapply(stack_df[, !(names(stack_df) %in% "Cancer")], 
                        function(x) as.numeric(as.character(x)))

#Now, since the outputs of the various models are extremely correlated let's use PCA to reduce the dimensionality of the dataset
pca_stack <- prcomp(numeric_st_df, scale = F)

# Transform the data into the principal components using the predict() fucntion and keep only 3 of the original components
predicted_stack <- as.data.frame(predict(pca_stack, numeric_st_df))[1:3]
# Now, add those columns to the target variable (Cancer) and convert it to a data frame
stacked_df <- data.frame(predicted_stack, Cancer = stack_df[, (names(stack_df) %in% "Cancer")])

#We will be building a linear SVM on the dataset to predict the final target variable
stacked_model <- svm(Cancer ~ . , stacked_df, kernel = "linear")

#store the predictions of the various models on the test data
svm_test <- predict(model_svm, test_data)
knn_test <- ifelse(predict(model_knn, test_data)[, 1] >  predict(model_knn, test_data)[, 2], 0, 1)
dt_test <- ifelse(predict(model_dt, test_data)[, 1] >  predict(model_dt, test_data)[, 2], 0, 1)
test_data$Cancer <- as.numeric(as.character(test_data$Cancer))

stack_df_test <- data.frame(svm = svm_test, knn = knn_test, tree = dt_test, Cancer = test_data$Cancer)
stack_df_test$Cancer <- as.factor(stack_df_test$Cancer)

# Convert all other variables into numeric
numeric_st_df_test <- sapply(stack_df_test[, !(names(stack_df_test) %in% "Cancer")],
                             function(x) as.numeric(as.character(x)))

# Apply dimensionality reduction on the numeric attributes
predicted_stack_test <- as.data.frame(predict(pca_stack, numeric_st_df_test))[1:3]

# Combine the target variable along with the reduced dataset
stacked_df_test <- data.frame(predicted_stack_test, Cancer = stack_df_test[, (names(stack_df_test) %in% "Cancer")])

#predictions of stacked model
preds_st_test <-  predict(stacked_model, stacked_df_test)

#Reporting metrics on test cases for all models

confusionMatrix(svm_test, test_data$Cancer)
confusionMatrix(knn_test, test_data$Cancer)
confusionMatrix(dt_test, test_data$Cancer)

confusionMatrix(preds_st_test, stacked_df_test$Cancer)




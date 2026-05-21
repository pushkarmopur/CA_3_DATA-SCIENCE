# CA 3 - Predictive Modelling

# 1. Loading the Required Libraries

library(psych)    
library(e1071)    
library(MASS)     
library(lmtest)# For Durbin-Watson test

install.packages("faraway")
install.packages("DAAG")

library(faraway)  # For VIF (multicollinearity)
library(DAAG)     # For K-fold cross validation



# 2. Loading the dataset

df <- read.csv("C:/Users/Pshkr/Downloads/student_performance_dataset.csv")



# Subset to the variables we need for the research question

my_data <- df[, c("exam_score", "study_hours", "sleep_hours", "focus_index", "mental_health_score")]



# Examine Initial Linearity

windows(20,10)

pairs.panels(my_data,
             
             smooth = FALSE,      
             
             scale = FALSE,       
             
             density = TRUE,      
             
             ellipses = FALSE,    
             
             method = "spearman", # Correlation method
             
             pch = 21,            
             
             lm = FALSE,          
             
             cor = TRUE,          
             
             jiggle = FALSE,      
             
             factor = 2,          
             
             hist.col = 4,        
             
             stars = TRUE,        
             
             ci = TRUE)           



# Seeing linearity in more detail using scatter plots

windows(20,12)

par(mfrow= c(2,2))



scatter.smooth(x = my_data$study_hours, y = my_data$exam_score,
               
               xlab = "Study Hours", ylab = "Exam Score", main = "Exam Score ~ Study Hours")



scatter.smooth(x = my_data$sleep_hours, y = my_data$exam_score,
               
               xlab = "Sleep Hours", ylab = "Exam Score", main = "Exam Score ~ Sleep Hours")



scatter.smooth(x = my_data$focus_index, y = my_data$exam_score,
               
               xlab = "Focus Index", ylab = "Exam Score", main = "Exam Score ~ Focus Index")



scatter.smooth(x = my_data$mental_health_score, y = my_data$exam_score,
               
               xlab = "Mental Health Score", ylab = "Exam Score", main = "Exam Score ~ Mental Health Score")



# Examining correlation 

cor(my_data)

paste("Correlation for Exam Score and Study Hours: ", round(cor(my_data$exam_score, my_data$study_hours),2))

paste("Correlation for Exam Score and Sleep Hours: ", round(cor(my_data$exam_score, my_data$sleep_hours),2))




# Outliers Checkinh


windows(20,10)

par(mfrow = c(2, 3)) 



boxplot(my_data$exam_score, main = "Exam Score")

boxplot(my_data$study_hours, main = "Study Hours")

boxplot(my_data$sleep_hours, main = "Sleep Hours")

boxplot(my_data$focus_index, main = "Focus Index")

boxplot(my_data$mental_health_score, main = "Mental Health Score")



# Identify and removing the sleep_hours outliers

outlier_values <- boxplot.stats(my_data$sleep_hours)$out

paste("Sleep Hours outliers: ", paste(outlier_values, sep =", "))

my_data <- subset(my_data, !(sleep_hours %in% outlier_values))


# Normality Checks & Transformations

windows(30,20)

par(mfrow = c(2,3)) 



plot(density(my_data$exam_score), main = "Density plot : Exam Score", ylab = "Frequency", xlab = "Exam Score", sub = paste("Skewness : ", round(skewness(my_data$exam_score), 2)))

polygon(density(my_data$exam_score), col = "red")



plot(density(my_data$study_hours), main = "Density plot : Study Hours", ylab = "Frequency", xlab = "Study Hours", sub = paste("Skewness : ", round(skewness(my_data$study_hours), 2)))

polygon(density(my_data$study_hours), col = "red")



plot(density(my_data$sleep_hours), main = "Density plot : Sleep Hours", ylab = "Frequency", xlab = "Sleep Hours", sub = paste("Skewness : ", round(skewness(my_data$sleep_hours), 2)))

polygon(density(my_data$sleep_hours), col = "red")



# Shapiro-Wilk Normality Test (using subset because limit is 5000)

shapiro.test(my_data$exam_score[1:4000]) 

shapiro.test(my_data$study_hours[1:4000]) 



# Box-Cox transformation for study_hours

box_cox_transform <- boxcox(my_data$exam_score ~ my_data$study_hours)

lamda <- box_cox_transform$x[which.max(box_cox_transform$y)]

my_data$study_hours_new <- (my_data$exam_score^lamda-1)/lamda




# Spliting the Data into Training and Testing



set.seed(123) 

no_of_records <- sample(1:nrow(my_data), 0.8 * nrow(my_data))

training_data <- my_data[no_of_records, ]

testing_data <- my_data[-no_of_records, ]




# Build & Validate Models

# Building Model 1 with All the Variables

model_1 <- lm(exam_score ~ study_hours_new + sleep_hours + focus_index + mental_health_score, data= training_data)

summary(model_1)



# Building Model 2 by Dropping weakest variable: sleep_hours

model_2 <- lm(exam_score ~ study_hours_new + focus_index + mental_health_score, data= training_data)



# Printing Summary Metrics for Model 2: which Includes Call, Residuals, Coefficients, R-Squared, F-Statistic

summary(model_2)



# Comparing the AIC and BIC

AIC(model_1)

AIC(model_2)

BIC(model_1)

BIC(model_2)


# Model Assumption Testing


# 1. Residuals normally distributed

shapiro.test(residuals(model_2)[1:4000])



# 2. Residuals mean equal to zero

t.test(residuals(model_2), mu=0) 



# 3. Autocorrelation (Durbin-Watson test)

dwtest(model_2)



# 4. Multicollinearity (VIF)

vif(model_2)


# K-Fold Cross Validation (DAAG)




windows(20,10)

cvResults <- suppressWarnings(CVlm(data = training_data, 
                                   
                                   form.lm = exam_score ~ study_hours_new + focus_index + mental_health_score, 
                                   
                                   m = 5, 
                                   
                                   dots = FALSE, 
                                   
                                   seed = 29, 
                                   
                                   legend.pos = "topleft",
                                   
                                   printit = FALSE, 
                                   
                                   main = "Small symbols are predicted values while bigger ones are actuals."))




# Model Forecasting & Evaluation


# Predicting  the scores using the testing data

predicted_scores <- predict(model_2, testing_data)



# Actual vs Predicted data frame

actuals_preds <- data.frame(cbind(
  
  actuals = testing_data$exam_score,
  
  predicted = predicted_scores
  
))



# View the actual vs predicted values

head(actuals_preds)



# 1. Correlation Accuracy

correlation_accuracy <- cor(actuals_preds$actuals, actuals_preds$predicted)

paste("Correlation Accuracy: ", round(correlation_accuracy, 4))



# 2. Mean Squared Error (MSE)

mse_value <- mean((actuals_preds$actuals - actuals_preds$predicted)^2)

paste("Mean Squared Error (MSE): ", round(mse_value, 2))



# 3. Mean Absolute Percentage Error (MAPE)

mape_value <- mean(abs((actuals_preds$actuals - actuals_preds$predicted) / actuals_preds$actuals)) * 100

paste("Mean Absolute Percentage Error (MAPE): ", round(mape_value, 2), "%")



# 4. Min-Max Accuracy

min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))

paste("Min-Max Accuracy: ", round(min_max_accuracy, 4))
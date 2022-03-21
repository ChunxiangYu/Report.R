#Report.R
#Anonymous marking code: Z0174573

install.packages("ggplot2")
install.packages("dplyr")
install.packages("e1071")
install.packages("caret")
install.packages("rpart")
install.packages("randomForest")
install.packages('magrittr')
install.packages("caTools")
install.packages("skimr")
install.packages("pROC")
library(ggplot2)
library(dplyr)  
library(e1071)     
library(caret)    
library(rpart)      
library(randomForest) 
library(magrittr)
library(caTools)
library(skimr) 
library(pROC)

hotel <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/hotels.csv",show_col_types = FALSE)

skimmed <- skim(hotel)
print(skimmed)

#Data Processing
numericFactor <- sapply(hotel, is.numeric)
y <- "is_canceled" 
x <- setdiff(names(hotel)[numericFactor], y)
cor(hotel[x], hotel[[y]])

hotel<-hotel%>%mutate(is_canceled=as.factor(is_canceled))
hotel[sapply(hotel, is.character)] <- lapply(hotel[sapply(hotel, is.character)], as.factor)

head(hotel)
#Divide train_set and test_set
set.seed(58)

train_index <- sample(1:nrow(hotel), 0.75 * nrow(hotel))
test_index <- setdiff(1:nrow(hotel), train_index)# set different index
training_set <- hotel[train_index, ]
test_set <- hotel[test_index, ]

training_set <- training_set[c('hotel','is_canceled','lead_time','adults','is_repeated_guest',
                               'previous_cancellations','previous_bookings_not_canceled','booking_changes',
                               'days_in_waiting_list','adr','required_car_parking_spaces',
                               'total_of_special_requests')]
test_set <- test_set[c('hotel','is_canceled','lead_time','adults','is_repeated_guest',
                       'previous_cancellations','previous_bookings_not_canceled','booking_changes',
                       'days_in_waiting_list','adr','required_car_parking_spaces',
                       'total_of_special_requests')]

p1 <- ggplot(hotel, aes(x= is_canceled,  group=arrival_date_month)) + 
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count",width = 1,position = position_dodge(1.2)) +
  geom_text(aes( label = scales::percent(..prop..),
                 y= ..prop.. ), stat= "count",size = 2.3,hjust=0.3) +
  labs(y = "Percent", x = "canceled",fill="cancel or not") +
  facet_grid(~arrival_date_month) +
  scale_y_continuous(labels = scales::percent)

print(p1)

p2<- ggplot(hotel, aes(x = adr, fill = hotel, color = hotel)) + 
  geom_histogram(aes(y = ..density..), position = position_dodge(), binwidth = 20 ) +
  geom_density(alpha = 0.2) + 
  labs(title = "Average Daily rate by Hotel",
       x = "Hotel Price(in Euro)",
       y = "Count") + scale_color_brewer(palette = "Paired") + 
  theme_classic() + theme(legend.position = "top")+ xlim(-1,400) 

print(p2)

p3 <-ggplot(hotel,aes(x = is_canceled ,y= previous_cancellations))+
  geom_bar(stat='summary',fun = 'mean',width = 0.7)+labs(title = "Percentage of customers who canceled a booking before canceling again")

print(p3)

p4 <- ggplot(hotel, mapping=aes(x="is_canceled",fill=is_canceled))+
  geom_bar(stat="count",width=0.5,position='stack')+coord_polar("y", start=0)+labs(title = "Percentage of actual check-ins and cancellations in hotel reservations")

print(p4)

p5 <-ggplot(hotel,aes(x = is_canceled ,y= is_repeated_guest))+
  geom_bar(stat='summary',fun = 'mean',width = 0.7)+labs(title = "Percentage of customers who canceled a booking is repeated guest")

print(p5)

without_fold_rf<-randomForest(is_canceled~.,    
                              data=training_set,          
                              ntree=500,                     
                              cutoff=c(0.5,0.5), 
                              mtry=2,
                              importance=TRUE)
print(without_fold_rf)

without_fold_rf_prob<-predict(without_fold_rf,test_set,type="prob")[,2]  
without_fold_rf_pred<-predict(without_fold_rf,test_set,type="class")
print(confusionMatrix(as.factor(without_fold_rf_pred),test_set$is_canceled))

#Build an initial random forest model from the selected features
initial_rf<-randomForest(is_canceled~.,    
                         data=training_set,          
                         ntree=500,                     
                         cutoff=c(0.5,0.5), 
                         mtry=2,
                         importance=TRUE,
                         cv.fold=10)
print(initial_rf)

initial_rf_prob<-predict(initial_rf,test_set,type="prob")[,2]  
initial_rf_pred<-predict(initial_rf,test_set,type="class")
print(confusionMatrix(as.factor(initial_rf_pred),test_set$is_canceled))

set.seed(58)              
best_mtry_model <- tuneRF(x = training_set%>%select(-is_canceled),
                          y = training_set$is_canceled,mtryStart=2,
                          ntreeTry = 500)
print(best_mtry_model)

new_rf<-randomForest(is_canceled~.,              
                     data=training_set,          
                     ntree=500,                     
                     cutoff=c(0.5,0.5), 
                     mtry=8,
                     importance=TRUE,
                     cv.fold=10)
print(new_rf)

rf_prob<-predict(new_rf,test_set,type="prob")[,2]  
rf_pred<-predict(new_rf,test_set,type="class")
print(confusionMatrix(as.factor(rf_pred),test_set$is_canceled))

ran_roc <- roc(test_set$is_canceled,rf_prob)
plot(ran_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='ROC curve,mtry=8,ntree=500')

dt_model<-rpart(is_canceled~.,
                data=training_set, 
                method="class",
                control=rpart.control(cp=0, maxdepth = 3))
dt_prob<-predict(dt_model,test_set)[,2]

log_model<-glm(is_canceled~.,family="binomial",data=training_set)
log_prob = predict(log_model,test_set,type="response")

rf_roc<-roc(test_set$is_canceled,rf_prob,auc=TRUE)

log_roc<-roc(test_set$is_canceled,log_prob,auc=TRUE)

dt_roc <- roc(test_set$is_canceled,dt_prob,auc=TRUE)

plot(rf_roc,print.auc=TRUE,print.auc.y=.1,col="red")
plot(dt_roc,print.auc=TRUE,print.auc.y=.2,col="yellow",add=TRUE)
plot(log_roc,print.auc=TRUE,print.auc.y=.3,col="blue",add=TRUE)

legend("left", legend = c("rf_roc", "dt_roc", "log_roc"),col = c("red", "yellow", "blue"),lwd = 2,horiz=TRUE)
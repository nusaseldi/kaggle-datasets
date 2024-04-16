

```r
# Load Library
library(tidymodels)
library(tidyverse)

# Import dataset
titanic_test <- read_csv("test.csv")
```

```
## Rows: 418 Columns: 11
## ── Column specification ───────────────────────────────────────────────────────────────────
## Delimiter: ","
## chr (5): Name, Sex, Ticket, Cabin, Embarked
## dbl (6): PassengerId, Pclass, Age, SibSp, Parch, Fare
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

```r
titanic_train <- read_csv("train.csv")
```

```
## Rows: 891 Columns: 12
## ── Column specification ───────────────────────────────────────────────────────────────────
## Delimiter: ","
## chr (5): Name, Sex, Ticket, Cabin, Embarked
## dbl (7): PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

```r
titanic_train |> summarise_all(~ sum(is.na(.)))
```

```
## # A tibble: 1 × 12
##   PassengerId Survived Pclass  Name   Sex   Age SibSp Parch Ticket  Fare Cabin Embarked
##         <int>    <int>  <int> <int> <int> <int> <int> <int>  <int> <int> <int>    <int>
## 1           0        0      0     0     0   177     0     0      0     0   687        2
```

```r
titanic_test |> summarise_all(~ sum(is.na(.)))
```

```
## # A tibble: 1 × 11
##   PassengerId Pclass  Name   Sex   Age SibSp Parch Ticket  Fare Cabin Embarked
##         <int>  <int> <int> <int> <int> <int> <int>  <int> <int> <int>    <int>
## 1           0      0     0     0    86     0     0      0     1   327        0
```

```r
titanic_full <- bind_rows(titanic_train, titanic_test)

titanic_full |> summarise_all(~ sum(is.na(.)))
```

```
## # A tibble: 1 × 12
##   PassengerId Survived Pclass  Name   Sex   Age SibSp Parch Ticket  Fare Cabin Embarked
##         <int>    <int>  <int> <int> <int> <int> <int> <int>  <int> <int> <int>    <int>
## 1           0      418      0     0     0   263     0     0      0     1  1014        2
```

```r
glimpse(titanic_full)
```

```
## Rows: 1,309
## Columns: 12
## $ PassengerId <dbl> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20…
## $ Survived    <dbl> 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, …
## $ Pclass      <dbl> 3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3, 2, 2, 3, …
## $ Name        <chr> "Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley (Florence Brig…
## $ Sex         <chr> "male", "female", "female", "female", "male", "male", "male", "male",…
## $ Age         <dbl> 22, 38, 26, 35, 35, NA, 54, 2, 27, 14, 4, 58, 20, 39, 14, 55, 2, NA, …
## $ SibSp       <dbl> 1, 1, 0, 1, 0, 0, 0, 3, 0, 1, 1, 0, 0, 1, 0, 0, 4, 0, 1, 0, 0, 0, 0, …
## $ Parch       <dbl> 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 5, 0, 0, 1, 0, 0, 0, 0, 0, 0, …
## $ Ticket      <chr> "A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450", "330…
## $ Fare        <dbl> 7.2500, 71.2833, 7.9250, 53.1000, 8.0500, 8.4583, 51.8625, 21.0750, 1…
## $ Cabin       <chr> NA, "C85", NA, "C123", NA, NA, "E46", NA, NA, NA, "G6", "C103", NA, N…
## $ Embarked    <chr> "S", "C", "S", "S", "S", "Q", "S", "S", "S", "C", "S", "S", "S", "S",…
```

```r
str(titanic_full)
```

```
## spc_tbl_ [1,309 × 12] (S3: spec_tbl_df/tbl_df/tbl/data.frame)
##  $ PassengerId: num [1:1309] 1 2 3 4 5 6 7 8 9 10 ...
##  $ Survived   : num [1:1309] 0 1 1 1 0 0 0 0 1 1 ...
##  $ Pclass     : num [1:1309] 3 1 3 1 3 3 1 3 3 2 ...
##  $ Name       : chr [1:1309] "Braund, Mr. Owen Harris" "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" "Heikkinen, Miss. Laina" "Futrelle, Mrs. Jacques Heath (Lily May Peel)" ...
##  $ Sex        : chr [1:1309] "male" "female" "female" "female" ...
##  $ Age        : num [1:1309] 22 38 26 35 35 NA 54 2 27 14 ...
##  $ SibSp      : num [1:1309] 1 1 0 1 0 0 0 3 0 1 ...
##  $ Parch      : num [1:1309] 0 0 0 0 0 0 0 1 2 0 ...
##  $ Ticket     : chr [1:1309] "A/5 21171" "PC 17599" "STON/O2. 3101282" "113803" ...
##  $ Fare       : num [1:1309] 7.25 71.28 7.92 53.1 8.05 ...
##  $ Cabin      : chr [1:1309] NA "C85" NA "C123" ...
##  $ Embarked   : chr [1:1309] "S" "C" "S" "S" ...
##  - attr(*, "spec")=
##   .. cols(
##   ..   PassengerId = col_double(),
##   ..   Survived = col_double(),
##   ..   Pclass = col_double(),
##   ..   Name = col_character(),
##   ..   Sex = col_character(),
##   ..   Age = col_double(),
##   ..   SibSp = col_double(),
##   ..   Parch = col_double(),
##   ..   Ticket = col_character(),
##   ..   Fare = col_double(),
##   ..   Cabin = col_character(),
##   ..   Embarked = col_character()
##   .. )
##  - attr(*, "problems")=<externalptr>
```

```r
# Data transformation
titanic_full <- titanic_full |>
  mutate(FamilySize = SibSp + Parch + 1) |>
  select(-SibSp, -Parch)

titanic_full$title <- titanic_full$Name |>
  str_extract("([A-z]+)\\.") |>
  str_sub(end = -2)
titanic_full |>
  group_by(title) |>
  count() |>
  arrange(desc(n)) |>
  ungroup() |>
  mutate(prop = n / sum(n))
```

```
## # A tibble: 18 × 3
##    title        n     prop
##    <chr>    <int>    <dbl>
##  1 Mr         757 0.578   
##  2 Miss       260 0.199   
##  3 Mrs        197 0.150   
##  4 Master      61 0.0466  
##  5 Dr           8 0.00611 
##  6 Rev          8 0.00611 
##  7 Col          4 0.00306 
##  8 Major        2 0.00153 
##  9 Mlle         2 0.00153 
## 10 Ms           2 0.00153 
## 11 Capt         1 0.000764
## 12 Countess     1 0.000764
## 13 Don          1 0.000764
## 14 Dona         1 0.000764
## 15 Jonkheer     1 0.000764
## 16 Lady         1 0.000764
## 17 Mme          1 0.000764
## 18 Sir          1 0.000764
```

```r
titanic_full <- titanic_full |> select(-Cabin, -Ticket, -Name)

titanic_full <- titanic_full |> mutate(across(c(Survived, Pclass, Sex, Embarked, title), as.factor))

# input missing data
impute_data <- recipe(Survived ~ ., data = titanic_full) |>
  step_impute_mode(Embarked) |>
  step_impute_linear(Fare) |>
  step_impute_knn(Age) |>
  prep()

imputed <- bake(impute_data, new_data = titanic_full)

imputed |> summarise_all(~ sum(is.na(.)))
```

```
## # A tibble: 1 × 9
##   PassengerId Pclass   Sex   Age  Fare Embarked FamilySize title Survived
##         <int>  <int> <int> <int> <int>    <int>      <int> <int>    <int>
## 1           0      0     0     0     0        0          0     0      418
```

```r
imputed <- imputed |> mutate(Age = case_when(
  Age < 8 ~ "anak-anak",
  Age >= 9 & Age < 19 ~ "remaja",
  Age >= 19 & Age <= 60 ~ "dewasa",
  TRUE ~ "lansia"
))

imputed$Age <- as.factor(imputed$Age)

glimpse(imputed)
```

```
## Rows: 1,309
## Columns: 9
## $ PassengerId <dbl> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20…
## $ Pclass      <fct> 3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3, 2, 2, 3, …
## $ Sex         <fct> male, female, female, female, male, male, male, male, female, female,…
## $ Age         <fct> dewasa, dewasa, dewasa, dewasa, dewasa, dewasa, dewasa, anak-anak, de…
## $ Fare        <dbl> 7.2500, 71.2833, 7.9250, 53.1000, 8.0500, 8.4583, 51.8625, 21.0750, 1…
## $ Embarked    <fct> S, C, S, S, S, Q, S, S, S, C, S, S, S, S, S, S, Q, S, S, C, S, S, Q, …
## $ FamilySize  <dbl> 2, 2, 1, 2, 1, 1, 1, 5, 3, 2, 3, 1, 1, 7, 1, 1, 6, 1, 2, 1, 1, 1, 1, …
## $ title       <fct> Mr, Mrs, Miss, Mrs, Mr, Mr, Mr, Master, Mrs, Mrs, Miss, Miss, Mr, Mr,…
## $ Survived    <fct> 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, …
```

```r
# build a model
titanic_train <- imputed |> slice(1:891)
titanic_test <- imputed |> slice(892:1309)

imputed |> view()

set.seed(777)
split <- initial_split(titanic_train, prop = 0.8, strata = Survived)
train <- training(split)
test <- testing(split)

titanic_recipe <- recipe(data = train, formula = Survived ~ .) |> 
  update_role(PassengerId, new_role = "id") |>
  step_normalize(all_numeric_predictors()) |>
  step_other(title, threshold = 0.02) |>
  step_dummy(all_nominal_predictors()) 
  
titanic_prep <- prep(titanic_recipe)

bake(titanic_prep, new_data = NULL)
```

```
## # A tibble: 712 × 16
##    PassengerId     Fare FamilySize Survived Pclass_X2 Pclass_X3 Sex_male Age_dewasa
##          <dbl>    <dbl>      <dbl> <fct>        <dbl>     <dbl>    <dbl>      <dbl>
##  1           5 -0.510       -0.566 0                0         1        1          1
##  2           7  0.454       -0.566 0                0         0        1          1
##  3           8 -0.223        1.86  0                0         1        1          0
##  4          13 -0.510       -0.566 0                0         1        1          1
##  5          14  0.00120      3.07  0                0         1        1          1
##  6          15 -0.514       -0.566 0                0         1        0          0
##  7          17 -0.0461       2.46  0                0         1        1          0
##  8          25 -0.223        1.86  0                0         1        0          0
##  9          27 -0.528       -0.566 0                0         1        1          1
## 10          28  5.10         2.46  0                0         0        1          1
## # ℹ 702 more rows
## # ℹ 8 more variables: Age_lansia <dbl>, Age_remaja <dbl>, Embarked_Q <dbl>,
## #   Embarked_S <dbl>, title_Miss <dbl>, title_Mr <dbl>, title_Mrs <dbl>, title_other <dbl>
```

```r
set.seed(345)
folds <- vfold_cv(train, v = 10, repeats = 1, strata = Survived)

rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) |> 
  set_engine("ranger") |>
  set_mode("classification")

titanic_workflow <- workflow() |>
  add_recipe(titanic_recipe) |>
  add_model(rf_spec)

rf_tune <- tune_grid(titanic_workflow,
  resamples = folds,
  grid = 10, 
  control = control_grid(save_pred = TRUE, parallel_over = "everything")
)
```

```
## i Creating pre-processing data to finalize unknown parameter: mtry
```

```r
# evaluate model
collect_metrics(rf_tune)
```

```
## # A tibble: 30 × 8
##     mtry min_n .metric     .estimator  mean     n std_err .config              
##    <int> <int> <chr>       <chr>      <dbl> <int>   <dbl> <chr>                
##  1    11     9 accuracy    binary     0.834    10 0.0163  Preprocessor1_Model01
##  2    11     9 brier_class binary     0.128    10 0.0103  Preprocessor1_Model01
##  3    11     9 roc_auc     binary     0.875    10 0.0165  Preprocessor1_Model01
##  4     9    25 accuracy    binary     0.834    10 0.0159  Preprocessor1_Model02
##  5     9    25 brier_class binary     0.126    10 0.00956 Preprocessor1_Model02
##  6     9    25 roc_auc     binary     0.872    10 0.0168  Preprocessor1_Model02
##  7    13    31 accuracy    binary     0.833    10 0.0171  Preprocessor1_Model03
##  8    13    31 brier_class binary     0.128    10 0.00975 Preprocessor1_Model03
##  9    13    31 roc_auc     binary     0.871    10 0.0161  Preprocessor1_Model03
## 10     6    13 accuracy    binary     0.844    10 0.0165  Preprocessor1_Model04
## # ℹ 20 more rows
```

```r
show_best(rf_tune, metric = "accuracy")
```

```
## # A tibble: 5 × 8
##    mtry min_n .metric  .estimator  mean     n std_err .config              
##   <int> <int> <chr>    <chr>      <dbl> <int>   <dbl> <chr>                
## 1     6    13 accuracy binary     0.844    10  0.0165 Preprocessor1_Model04
## 2     6    37 accuracy binary     0.838    10  0.0162 Preprocessor1_Model08
## 3    13    35 accuracy binary     0.837    10  0.0162 Preprocessor1_Model09
## 4     8    25 accuracy binary     0.836    10  0.0164 Preprocessor1_Model07
## 5     9    25 accuracy binary     0.834    10  0.0159 Preprocessor1_Model02
```

```r
final_rf <- titanic_workflow |>
  finalize_workflow(select_best(rf_tune, metric = "accuracy"))

final_fit <- final_rf |>
  last_fit(split)

collect_metrics(final_fit)
```

```
## # A tibble: 3 × 4
##   .metric     .estimator .estimate .config             
##   <chr>       <chr>          <dbl> <chr>               
## 1 accuracy    binary         0.849 Preprocessor1_Model1
## 2 roc_auc     binary         0.884 Preprocessor1_Model1
## 3 brier_class binary         0.116 Preprocessor1_Model1
```

```r
final_model <- extract_workflow(final_fit)

pred <- final_model |> predict(titanic_test)
pred
```

```
## # A tibble: 418 × 1
##    .pred_class
##    <fct>      
##  1 0          
##  2 1          
##  3 0          
##  4 0          
##  5 1          
##  6 0          
##  7 1          
##  8 0          
##  9 1          
## 10 0          
## # ℹ 408 more rows
```

```r
prediksi <- titanic_test |>
  select(PassengerId) |>
  bind_cols(pred) |>
  rename(Survived = .pred_class)

write_csv(prediksi, file = "predict.csv")
```


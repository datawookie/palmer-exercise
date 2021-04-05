# -----------------------------------------------------------------------------
#
# TIDYMODELS: Palmer Penguins
#
# -----------------------------------------------------------------------------

library(dplyr)
library(ggplot2)
library(ggsci)
library(tidymodels)
library(ranger)
library(kernlab)

# To install the Palmer Penuins library:
#
# > install.packages("palmerpenguins")

# Load the library.
#
library(palmerpenguins)

# Take a quick look at the data:
#
glimpse(penguins)
#
# To access the documentation:
#
# > ?penguins

# PREPARE ---------------------------------------------------------------------

# We'll take the cowardly approach and just remove the two records with missing
# data.
#
penguins <- na.omit(penguins)

# EDA -------------------------------------------------------------------------

# Plot body mass versus flipper length.
#
# What do you observe? Will these features be useful for classification?
#
ggplot(data = penguins, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point(aes(color = species), size = 2) +
  scale_color_jco() +
  labs(
    x = "Flipper Length",
    y = "Body Mass"
  )

# Plot bill depth versus bill length.
#
# What do you observe? Will these features be useful for classification?
#
ggplot(data = penguins, aes(x = bill_length_mm, y = bill_depth_mm)) +
  geom_point(aes(color = species), size = 2) +
  scale_color_jco() +
  labs(
    x = "Bill Length",
    y = "Bill Depth"
  )

# RSAMPLE ---------------------------------------------------------------------

# Split the data:
#
# - 80% → training
# - 20% → testing
#
# Use the target column for stratification.
#
splits <- initial_split(penguins, prop = 0.8, strata = species)

train <- training(splits) 
test  <- testing(splits)

# RECIPE ----------------------------------------------------------------------

penguins_cooked <- recipe(
  species ~ .,
  data = train
) %>%
  # Centre all *numeric* features.
  step_center(all_numeric()) %>%
  # Scale all *numeric* features.
  step_scale(all_numeric()) %>%
  # Remove constant features.
  step_zv(all_predictors())

# PARSNIP ---------------------------------------------------------------------

# Create a Decision Tree model.
#
# penguins_tree <- decision_tree() %>%
#   set_engine("rpart") %>% 
#   set_mode("classification")

penguins_tree <- svm_rbf() %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# WORKFLOW --------------------------------------------------------------------

work <- workflow() %>%
  add_model(penguins_tree) %>% 
  add_recipe(penguins_cooked)

# Fit the model to the training data.
#
model_fit <- work %>% 
  fit(train)

# Make predictions and add the reference values.
#
predictions <- predict(model_fit, test) %>% 
  bind_cols(test %>% select(species))

# YARDSTICK -------------------------------------------------------------------

predictions %>%                  
  accuracy(truth = species, .pred_class)

# Add Cross Validation to the workflow.
#
folds <- vfold_cv(train, v = 10)
#
work <- work %>% 
  fit_resamples(folds)

collect_metrics(work)

# EXERCISES -------------------------------------------------------------------

# 1. Try some different models:
#
#    - Random Forest          | rand_forest() | engine = "ranger"
#    - Support Vector Machine | svm_rbf()     | engine = "kernlab"
#
# 2. Refine the Cross Validation:
#
#    - repeat 5 times and
#    - stratify the samples using the target column.
#
# 3. Change the target column from 'species' to 'sex'.

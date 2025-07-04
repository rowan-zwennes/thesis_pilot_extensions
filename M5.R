# Ensure other libraries are installed and loaded
# install.packages("caret")        
# install.packages("RWeka")        
# install.packages("bestNormalize")
# install.packages("dplyr")        
# install.packages("purrr")       
# install.packages("jsonlite")    
# install.packages("tools")  

library(caret) 
library(RWeka)
library(bestNormalize)
library(dplyr)
library(purrr) 
library(jsonlite)
library(tools) 

# java_home_path = "C:/Program Files/Java/jdk-24" # <-- Change path here
# Sys.setenv(JAVA_HOME = java_home_path)

# Function to replicate Python's _load_other_data preprocessing in R
#' Preprocess R Data to Mimic Python's _load_other_data Function
#'
#' This function reads a CSV file, processes missing values, converts character columns 
#' to numeric or factors, removes problematic rows/columns, and separates features from 
#' the target. It is designed to mirror the preprocessing behavior of the Python data loader.
#'
#' @param csv_file_path A string. The file path to the CSV data file.
#' @param target_column_name A string. The name of the target column in the dataset. Default is "Target".
#'
#' @return A list with the following components:
#' \describe{
#'   \item{processed_data}{A cleaned data.frame with features and target column.}
#'   \item{target_col_name}{The name of the target column.}
#'   \item{cat_names}{A character vector of column names identified as categorical.}
#'   \item{rows_removed_count}{Number of rows removed due to missing values.}
#'   \item{cols_removed_count}{Number of columns removed due to high NA percentage.}
#'   \item{original_rows}{Original number of rows before cleaning.}
#'   \item{original_cols}{Original number of feature columns (excluding the target).}
#' }
#'
#' @export
preprocess_r_data <- function(csv_file_path, target_column_name = "Target") {
  cat("Reading data from:", csv_file_path, "\n")
  # Read CSV, ensure strings are not factors yet 
  data <- read.csv(csv_file_path, stringsAsFactors = FALSE, na.strings = c("NA", "", "?"))
  original_rows <- nrow(data)
  original_cols <- ncol(data)
  
  # Ensure target column exists
  if (!target_column_name %in% colnames(data)) {
    stop(paste("Target column '", target_column_name, "' not found in the data."), call. = FALSE)
  }
  
  # Separate X and y
  y_vec <- data[[target_column_name]]
  X_df <- data[, setdiff(colnames(data), target_column_name), drop = FALSE]
  
  cat("Initial X dimensions:", paste(dim(X_df), collapse = "x"), "\n")
  
  # Convert target to numeric
  if (!is.numeric(y_vec)) {
    y_vec <- as.numeric(as.character(y_vec))
  }
  
  # Handle missing values
  rows_removed <- 0
  cols_removed <- 0
  
  X_df_processed <- X_df
  for (col_name in names(X_df_processed)) {
    col_data <- X_df_processed[[col_name]]
    
    # First, try to convert to numeric if it's character
    if (is.character(col_data)) {
      suppressWarnings({
        num_conversion <- as.numeric(col_data)
      })
      
      # Check if conversion was successful (less than 50% became NA)
      na_before <- sum(is.na(col_data))
      na_after <- sum(is.na(num_conversion))
      
      if ((na_after - na_before) / length(col_data) < 0.5) {
        # Successful numeric conversion
        X_df_processed[[col_name]] <- num_conversion
      } else {
        # Keep as character (will be converted to factor later)
        X_df_processed[[col_name]] <- col_data
      }
    }
    # If already numeric, keep as is
  }
  
  # Row indices before any NA processing on X_df_processed and y_vec
  valid_initial_indices <- 1:nrow(X_df_processed)
  
  # Remove columns with >50% NAs in X
  na_col_proportions <- colMeans(is.na(X_df_processed))
  cols_to_remove_names <- names(na_col_proportions[na_col_proportions > 0.5])
  if (length(cols_to_remove_names) > 0) {
    X_df_processed <- X_df_processed[, !colnames(X_df_processed) %in% cols_to_remove_names, drop = FALSE]
    cols_removed <- length(cols_to_remove_names)
    cat("Removed", cols_removed, "columns with >50% NAs:", paste(cols_to_remove_names, collapse=", "), "\n")
  }
  
  # Identify rows with NAs in remaining X or in y
  na_in_y_indices <- which(is.na(y_vec))
  na_in_X_indices <- which(apply(is.na(X_df_processed), 1, any))
  rows_to_remove_indices <- unique(c(na_in_y_indices, na_in_X_indices))
  
  # Remove rows with NA value
  if (length(rows_to_remove_indices) > 0) {
    X_df_processed <- X_df_processed[-rows_to_remove_indices, , drop = FALSE]
    y_vec <- y_vec[-rows_to_remove_indices]
    rows_removed <- length(rows_to_remove_indices)
    cat("Removed", rows_removed, "rows with any NAs.\n")
  }
  
  cat("Processed X dimensions:", paste(dim(X_df_processed), collapse = "x"), "\n")
  cat(nrow(X_df_processed), "rows and", ncol(X_df_processed), "columns remaining.\n")
  
  # Identify categorical features
  cat_names <- c()
  for (col_name in names(X_df_processed)) {
    col_data <- X_df_processed[[col_name]]
    
    if (is.character(col_data)) {
      # Character columns are categorical
      X_df_processed[[col_name]] <- as.factor(col_data)
      cat_names <- c(cat_names, col_name)
      cat("  Identified '", col_name, "' as categorical (was character).\n")
    } else if (is.numeric(col_data)) {
      # Check if numeric column has few unique values 
      n_unique <- length(unique(col_data[!is.na(col_data)]))
      if (n_unique < 5 && n_unique > 1) {
        X_df_processed[[col_name]] <- as.factor(col_data)
        cat_names <- c(cat_names, col_name)
        cat("  Identified '", col_name, "' as categorical (few unique numeric values).\n")
      }
    }
  }
  
  # Combine processed X and y back into a single data frame for M5P
  final_data <- X_df_processed
  final_data[[target_column_name]] <- y_vec
  
  return(list(
    processed_data = final_data,
    target_col_name = target_column_name,
    cat_names = cat_names,
    rows_removed_count = rows_removed,
    cols_removed_count = cols_removed,
    original_rows = original_rows,
    original_cols = original_cols - 1 # -1 for target
  ))
}

#' Run Nested Cross-Validation for M5P with Yeo-Johnson Transformation
#'
#' This function performs nested cross-validation for the M5P regression model using
#' the `RWeka::M5P()` function. It includes an optional Yeo-Johnson transformation for 
#' numeric predictors (via `bestNormalize::yeojohnson()`) and robust handling of factor levels 
#' between training and test folds.
#'
#' @param full_data A data frame containing all the data (used if `data_set != "real"`).
#' @param target_col_name A string indicating the name of the target column to predict.
#' @param outer_fold_indices_list A list of lists, each containing `train` and `test` indices for outer CV.
#' @param full_data_list A list of pre-split training and test sets (used only if `data_set == "real"`).
#' @param m_vals A numeric vector of `M` values to tune during inner cross-validation (default: `c(1, 5, 10, 20)`).
#' @param inner_k Number of folds to use for inner cross-validation (default: 5).
#' @param random_seed Seed for reproducibility (default: 123).
#' @param data_set String indicating whether the dataset is "real" or simulated. Determines which data input is used.
#'
#' @return A list with two elements:
#' \describe{
#'   \item{results_per_fold}{A data frame with outer fold results: `Fold`, `Best_M`, and `M5P_MSE`.}
#'   \item{mean_mse}{The mean MSE across all outer folds (excluding failed folds).}
#' }
#'
#' @importFrom RWeka M5P Weka_control
#' @importFrom bestNormalize yeojohnson
#' @importFrom caret createFolds
#'
#' @export
run_m5p_nested_cv_yj <- function(full_data = NULL, 
                                 target_col_name, 
                                 outer_fold_indices_list = NULL,
                                 full_data_list = NULL,
                                 m_vals = c(1, 5, 10, 20), 
                                 inner_k = 5, 
                                 random_seed = 123,
                                 data_set) {
  
  # Check if there are NA values in target column
  if (data_set != "real") {
    if (any(is.na(full_data[[target_col_name]]))) {
      stop("NAs found in target column '", target_col_name, "'. Please handle them.", call. = FALSE)
    }
  }
  
  set.seed(random_seed)
  
  # Create empty dataframe for results and make m5 formula
  outer_cv_results <- data.frame(Fold = integer(), Best_M = integer(), M5P_MSE = numeric())
  formula_str <- paste(target_col_name, "~ .")
  m5p_formula <- as.formula(formula_str)
  
  # Loop over the (outer) folds
  num_outer_folds <- length(outer_fold_indices_list)
  for (i in 1:num_outer_folds) {
    cat("\n===== Outer Fold", i, "of", num_outer_folds, "(Applying Yeo-Johnson Transformation) =====\n")
    
    # Apply Yeo-Johnson and get the correct fold indices
    if (data_set != "real") {
      train_idx <- outer_fold_indices_list[[i]]$train
      test_idx <- outer_fold_indices_list[[i]]$test
      
      current_train_set <- full_data[train_idx, ]
      current_test_set <- full_data[test_idx, ]
      
      # --- YEO-JOHNSON TRANSFORMATION ---
      predictor_cols <- setdiff(colnames(current_train_set), target_col_name)
      
      current_train_set_for_model <- current_train_set
      current_test_set_for_model <- current_test_set
      
      if (length(predictor_cols) > 0) {
        cat("  Applying Yeo-Johnson transformation to NUMERIC predictors for this fold.\n")
        yj_transformers <- list()
        
        for (col_name in predictor_cols) {
          if (is.factor(current_train_set_for_model[[col_name]])) {
            cat("    Predictor '", col_name, "' is a factor. Skipping Yeo-Johnson transformation.\n")
            
            # Factor level handling
            train_levels <- levels(current_train_set_for_model[[col_name]])
            test_col_data <- current_test_set_for_model[[col_name]]
            
            if (!is.factor(test_col_data)) {
              # Convert test column to factor with training levels
              current_test_set_for_model[[col_name]] <- factor(
                as.character(test_col_data), 
                levels = train_levels
              )
            } else {
              # Test column is already factor, but ensure same levels
              test_levels <- levels(test_col_data)
              
              # Check for new levels in test set
              new_levels <- setdiff(as.character(test_col_data), train_levels)
              if (length(new_levels) > 0) {
                cat("    WARNING: Test set has new factor levels for '", col_name, "': ", 
                    paste(new_levels, collapse=", "), ". Setting to NA.\n")
                
                # Convert test values not in training levels to NA
                test_values <- as.character(test_col_data)
                test_values[!test_values %in% train_levels] <- NA
                current_test_set_for_model[[col_name]] <- factor(test_values, levels = train_levels)
              } else {
                # Reorder levels to match training set
                current_test_set_for_model[[col_name]] <- factor(
                  as.character(test_col_data), 
                  levels = train_levels
                )
              }
            }
            next 
          }
          
          # More type checks and conversions
          if (!is.numeric(current_train_set_for_model[[col_name]])) {
            cat("    WARNING: Predictor '", col_name, "' is neither factor nor numeric. Attempting coercion.\n")
            
            # Try to convert both train and test
            train_col_original <- current_train_set_for_model[[col_name]]
            test_col_original <- current_test_set_for_model[[col_name]]
            
            current_train_set_for_model[[col_name]] <- as.numeric(as.character(train_col_original))
            current_test_set_for_model[[col_name]] <- as.numeric(as.character(test_col_original))
            
            # Check for introduced NAs 
            if (any(is.na(current_train_set_for_model[[col_name]])) && !any(is.na(train_col_original))) {
              cat("      WARNING: NAs introduced in TRAINING for '", col_name, "' during coercion.\n")
            }
            if (any(is.na(current_test_set_for_model[[col_name]])) && !any(is.na(test_col_original))) {
              cat("      WARNING: NAs introduced in TEST for '", col_name, "' during coercion.\n")
            }
            
            if (!is.numeric(current_train_set_for_model[[col_name]])) {
              cat("    ERROR: Could not convert '", col_name, "' to numeric. Skipping YJ transformation.\n")
              next 
            }
          }
          
          # Check if column has zero variance (if so, skip)
          train_col_values_numeric <- current_train_set_for_model[[col_name]]
          unique_numeric_vals <- unique(train_col_values_numeric[!is.na(train_col_values_numeric)])
          if (length(unique_numeric_vals) <= 1) {
            cat("    WARNING: Numeric predictor '", col_name, "' has zero variance. Skipping YJ transformation.\n")
            next 
          }
          
          # Do the actual transformation
          yj_obj <- NULL 
          tryCatch({
            if(all(is.na(current_train_set_for_model[[col_name]]))) {
              cat("    WARNING: Numeric predictor '", col_name, "' is all NAs. Skipping YJ transformation.\n")
            } else {
              yj_obj <- bestNormalize::yeojohnson(current_train_set_for_model[[col_name]], standardize = TRUE)
            }
          }, error = function(e) {
            cat("    ERROR fitting Yeo-Johnson for numeric predictor '", col_name, "': ", e$message, ". Skipping transformation.\n")
          })
          
          # Use the same transformer for train and test datasets
          if (!is.null(yj_obj)) {
            yj_transformers[[col_name]] <- yj_obj
            current_train_set_for_model[[col_name]] <- predict(yj_obj, newdata = current_train_set_for_model[[col_name]])
            current_test_set_for_model[[col_name]] <- predict(yj_obj, newdata = current_test_set_for_model[[col_name]])
          }
        } 
      } else {
        cat("  No predictor columns found to transform. Using original data.\n")
      }
      
      # If dataset is Real Estate, the transformed data should be imported from python
    } else {
      current_train_set_for_model <- full_data_list[[i]]$train_set
      current_test_set_for_model  <- full_data_list[[i]]$test_set
    }
    
    # --- Inner CV (uses current_train_set_for_model which is now YJ transformed) ---
    inner_folds <- createFolds(current_train_set_for_model[[target_col_name]], k = inner_k, list = TRUE, returnTrain = FALSE)
    inner_cv_results_fold <- data.frame(M = integer(), Fold = integer(), MSE = numeric())
    
    cat("  Starting Inner", inner_k, "-fold CV for M tuning (on YJ transformed data)...\n")
    # Loop over each hyperparameter value
    for (m_param in m_vals) {
      fold_mses_for_m <- numeric(inner_k)
      
      # Loop over each inner fold
      for (j in 1:inner_k) {
        val_idx_inner <- inner_folds[[j]]
        
        inner_val_set <- current_train_set_for_model[val_idx_inner, ]
        inner_train_set <- current_train_set_for_model[-val_idx_inner, ]
        
        m5p_model_inner <- NULL
        preds_m5p_inner <- NULL
        
        # Try to make the model
        m5p_model_inner <- tryCatch({
          M5P(m5p_formula, data = inner_train_set, control = Weka_control(M = m_param))
        }, error = function(e) { return(NULL) })
        
        # Try to predict with the model
        if (!is.null(m5p_model_inner)) {
          preds_m5p_inner <- tryCatch({
            predict(m5p_model_inner, newdata = inner_val_set)
          }, error = function(e) { return(NULL) })
          
          if (!is.null(preds_m5p_inner)) {
            if (is.factor(preds_m5p_inner)) preds_m5p_inner <- as.numeric(as.character(preds_m5p_inner))
            else if (!is.numeric(preds_m5p_inner)) preds_m5p_inner <- as.numeric(preds_m5p_inner)
            
            # Calculate MSE
            if (is.numeric(preds_m5p_inner) && is.numeric(inner_val_set[[target_col_name]])) {
              fold_mses_for_m[j] <- mean((preds_m5p_inner - inner_val_set[[target_col_name]])^2, na.rm = TRUE)
            } else { fold_mses_for_m[j] <- Inf }
          } else { fold_mses_for_m[j] <- Inf }
        } else { fold_mses_for_m[j] <- Inf }
      }
      inner_cv_results_fold <- rbind(inner_cv_results_fold, data.frame(M = m_param, Fold = 1:inner_k, MSE = fold_mses_for_m))
    }
    
    # Determine best hyperparameter
    best_M_for_fold <- 4 
    if (nrow(inner_cv_results_fold) > 0 && !all(is.infinite(inner_cv_results_fold$MSE))) {
      avg_mse_per_m <- aggregate(MSE ~ M, data = inner_cv_results_fold, mean, na.action = na.omit)
      if (nrow(avg_mse_per_m) > 0 && sum(!is.na(avg_mse_per_m$MSE)) > 0) {
        best_M_for_fold <- avg_mse_per_m$M[which.min(avg_mse_per_m$MSE)]
      } else { cat("  WARNING: Inner CV yielded no valid MSEs after aggregation or all were NA. Using default M=4.\n") }
    } else { cat("  WARNING: Inner CV had errors or Inf values for all M values. Using default M=4.\n") }
    cat("  Best M from inner CV:", best_M_for_fold, "\n")
    
    # --- Train final M5P model ---
    final_m5p_model_outer <- NULL
    final_preds_m5p_outer <- NULL
    final_mse_m5p_outer <- Inf
    
    final_m5p_model_outer <- tryCatch({
      M5P(m5p_formula, data = current_train_set_for_model, control = Weka_control(M = best_M_for_fold))
    }, error = function(e) { 
      cat("ERROR training FINAL M5P (M=", best_M_for_fold, "): ", e$message, "\n")
      return(NULL) 
    })
    
    # Predict with final model
    if (!is.null(final_m5p_model_outer)) {
      final_preds_m5p_outer <- tryCatch({
        predict(final_m5p_model_outer, newdata = current_test_set_for_model)
      }, error = function(e) { 
        cat("ERROR predicting FINAL M5P (M=", best_M_for_fold, "): ", e$message, "\n")
        return(NULL) 
      })
      
      # Calculate MSE
      if (!is.null(final_preds_m5p_outer)) {
        if (is.factor(final_preds_m5p_outer)) final_preds_m5p_outer <- as.numeric(as.character(final_preds_m5p_outer))
        else if (!is.numeric(final_preds_m5p_outer)) final_preds_m5p_outer <- as.numeric(final_preds_m5p_outer)
        
        if (is.numeric(final_preds_m5p_outer) && is.numeric(current_test_set_for_model[[target_col_name]])) {
          final_mse_m5p_outer <- mean((final_preds_m5p_outer - current_test_set_for_model[[target_col_name]])^2, na.rm = TRUE)
        }
      }
    }
    cat("  Outer Fold M5P MSE (YJ Transformed):", final_mse_m5p_outer, "\n")
    
    outer_cv_results <- rbind(outer_cv_results, data.frame(Fold = i, Best_M = best_M_for_fold, M5P_MSE = final_mse_m5p_outer))
  } # End of outer for loop (i)
  
  cat("\n===== Final Nested CV Results (M5P, Yeo-Johnson Transformed) =====\n")
  print(outer_cv_results)
  
  # Calculate the average MSE from the (outer) folds
  mean_m5p_mse <- NA
  if (nrow(outer_cv_results) > 0 && !all(is.infinite(outer_cv_results$M5P_MSE))) {
    mean_m5p_mse <- mean(outer_cv_results$M5P_MSE, na.rm = TRUE)
    cat("Mean M5P MSE across outer folds (YJ Transformed):", mean_m5p_mse, "\n")
  } else {
    cat("Mean M5P MSE (YJ Transformed): Not calculable due to errors or Inf values.\n")
  }
  
  return(list(results_per_fold = outer_cv_results, mean_mse = mean_m5p_mse))
}

# --- Configuration ---
DATASET_CSV_DIR <- "C:/Users/rowan/Documents/" # <-- Change path here
JSON_FOLDS_DIR <- "C:/Users/rowan/Downloads/" # <-- Change path here
OUTPUT_RESULTS_FILE <- "C:/Users/rowan/Documents/mp5_output_final.csv" # <-- Change path here

# Available datasets: "rescosts", "thermF", "admission", "airfoil", "communities", "ozone", "boston", "slumpFL", "real", "tecator", "music"
DATASETS_TO_RUN <- c("rescosts", "thermF", "admission", "airfoil", "communities", "ozone", "boston", "slumpFL", "real", "tecator", "music") # Base names
MODELS_TO_RUN <- c("M5P_YJ") # For now, just M5P. Add others later.
TARGET_COLUMN_NAME <- "Target"

# --- Main Benchmarking Loop ---
all_results_data <- list()

# Loop over each selected dataset
for (base_name in DATASETS_TO_RUN) {
    cat(paste("\n===== Processing Base Dataset:", base_name, "=====\n"))
    
    # A. Construct & Verify CSV file path
    csv_filename <- paste0(base_name, "_table.csv")
    csv_file_path <- file.path(DATASET_CSV_DIR, csv_filename)
    if (!file.exists(csv_file_path)) {
      cat(paste("  WARNING: CSV '", csv_file_path, "' not found. Skipping.\n"))
      next
    }
    cat(paste("  Using CSV:", csv_file_path, "\n"))
    
    # B. Global Preprocessing
    preprocessed_output <- NULL
    tryCatch({
      preprocessed_output <- preprocess_r_data(csv_file_path, target_column_name = TARGET_COLUMN_NAME)
    }, error = function(e) {
      cat(paste("  ERROR preprocessing", base_name, ":", e$message, "\n"))
    })
    if (is.null(preprocessed_output)) next
    globally_preprocessed_data <- preprocessed_output$processed_data
    actual_target_name <- preprocessed_output$target_col_name
    
    
    # C. Construct & Verify JSON Fold file path
    json_filename <- paste0(base_name, "_fold_indices.json")
    json_fold_file_path <- file.path(JSON_FOLDS_DIR, json_filename)
    if (!file.exists(json_fold_file_path)) {
      cat(paste("  WARNING: JSON '", json_fold_file_path, "' not found. Skipping.\n"))
      next
    }
    cat(paste("  Using JSON folds:", json_fold_file_path, "\n"))
    
    # Load and Convert 0-based JSON Fold Indices to 1-based R list
    outer_fold_indices_list_from_json_0based <- NULL
    tryCatch({
      outer_fold_indices_list_from_json_0based <- jsonlite::fromJSON(json_fold_file_path)
    }, error = function(e) {
      cat(paste("  ERROR reading JSON '", json_fold_file_path, "': ", e$message, "\n"))
    })
    if (is.null(outer_fold_indices_list_from_json_0based)) next
    
    # Make the train and test sets 
    if(is.data.frame(outer_fold_indices_list_from_json_0based)) {
      temp_list_folds <- list()
      for(k_row in 1:nrow(outer_fold_indices_list_from_json_0based)){
        temp_list_folds[[k_row]] <- as.list(outer_fold_indices_list_from_json_0based[k_row, ])
      }
      outer_fold_indices_list_from_json_0based <- temp_list_folds
    }
    outer_fold_indices_list_1based <- lapply(outer_fold_indices_list_from_json_0based, function(fold_item) {
      train_key <- if ("train_indices_0based" %in% names(fold_item)) "train_indices_0based" else "train_indices"
      test_key <- if ("test_indices_0based" %in% names(fold_item)) "test_indices_0based" else "test_indices"
      if (!(train_key %in% names(fold_item) && test_key %in% names(fold_item))) {
        stop(paste0("JSON fold item for '", base_name, "' missing keys. Found: ", paste(names(fold_item), collapse=", ")))
      }
      list(train = as.numeric(unlist(fold_item[[train_key]])) + 1, 
           test = as.numeric(unlist(fold_item[[test_key]])) + 1)
    })
    cat(paste("  Loaded", length(outer_fold_indices_list_1based), "folds.\n"))
    
  # For Real Estate dataset use the from python imported datasets
  if (base_name == "real") { 
    
    cat(paste("\n===== Processing SPECIAL Dataset:", base_name, "(Loading from Python preprocessed JSON) =====\n"))
    
    # A. Construct & Verify JSON file path for the "real" dataset's preprocessed data
    real_data_json_filename <- paste0(base_name, "_label_encoded_datasets.json")
    real_data_json_file_path <- file.path(JSON_FOLDS_DIR, real_data_json_filename) # Ensure JSON_FOLDS_DIR is correct
    
    if (!file.exists(real_data_json_file_path)) {
      cat(paste("  WARNING: Preprocessed JSON data file '", real_data_json_file_path, "' for '", base_name, "' not found. Skipping this dataset.\n"))
      # If you skip here, you need a way for the outer loop (that runs models) to know this dataset has no data
      # For now, we'll assume it's found. If not, processed_real_data_folds will remain empty or NULL.
      next # Skip to the next dataset in DATASETS_TO_RUN
    }
    cat(paste("  Using preprocessed JSON data from:", real_data_json_file_path, "\n"))
    
    # B. Load the entire list of folds from the JSON file
    all_folds_data_from_json <- NULL
    tryCatch({
      all_folds_data_from_json <- jsonlite::fromJSON(real_data_json_file_path, simplifyDataFrame = FALSE)
    }, error = function(e) {
      cat(paste("  ERROR reading preprocessed JSON data '", real_data_json_file_path, "': ", e$message, "\n"))
    })
    
    # Create a list to store the final data.frames for each fold of the "real" dataset
    real_dataset_prepared_folds_list <- list()
    
    if (!is.null(all_folds_data_from_json) && is.list(all_folds_data_from_json) && length(all_folds_data_from_json) > 0) {
      cat(paste("  Loaded data for", length(all_folds_data_from_json), "folds from JSON. Reconstructing and processing...\n"))
      
      # Loop over each (outer) fold
      for (fold_idx_json in 1:length(all_folds_data_from_json)) {
        current_fold_json_data <- all_folds_data_from_json[[fold_idx_json]]
        fold_number_from_json <- current_fold_json_data$fold

        
        # --- Reconstruct Training Set X ---
        X_train_fold_current <- NULL
        if (!is.null(current_fold_json_data$train_set_data) && length(current_fold_json_data$train_set_data) > 0) {
          X_train_fold_current = current_fold_json_data$train_set_data
          X_train_fold_current = as.data.frame(X_train_fold_current)
        
          if (!is.null(X_train_fold_current) && !is.null(current_fold_json_data$train_column_names) && ncol(X_train_fold_current) == length(current_fold_json_data$train_column_names)) {
            colnames(X_train_fold_current) <- current_fold_json_data$train_column_names
          }
        }
        
        fold_indices_train = outer_fold_indices_list_1based[[fold_idx_json]]$train
        current_train_data_set = globally_preprocessed_data[fold_indices_train,]
        X_train_fold_current[["Target"]] = current_train_data_set[["Target"]]
        
        
        if (is.null(X_train_fold_current)) {
          cat("    ERROR: Could not reconstruct train_set_data for fold", fold_number_from_json, ". Skipping this fold.\n")
          next
        }
        
        # --- Reconstruct Test Set X ---
        X_test_fold_current <- NULL
        if (!is.null(current_fold_json_data$test_set_data) && length(current_fold_json_data$test_set_data) > 0) {
          X_test_fold_current = current_fold_json_data$test_set_data
          X_test_fold_current = as.data.frame(X_test_fold_current)
          
          if (!is.null(X_test_fold_current) && !is.null(current_fold_json_data$test_column_names) && ncol(X_test_fold_current) == length(current_fold_json_data$test_column_names)) {
            colnames(X_test_fold_current) <- current_fold_json_data$test_column_names
          }
        }
        if (is.null(X_test_fold_current)) {
          cat("    ERROR: Could not reconstruct test_set_data for fold", fold_number_from_json, ". Skipping this fold.\n")
          next
        }
        
        fold_indices_test = outer_fold_indices_list_1based[[fold_idx_json]]$test
        current_test_data_set = globally_preprocessed_data[fold_indices_test,]
        X_test_fold_current[["Target"]] = current_test_data_set[["Target"]]
        
        # --- Get Categorical Column Names ---
        cat_names_for_fold <- current_fold_json_data$categorical_names
        
        # --- Convert Specified Categorical Columns to Factors ---
        if (!is.null(cat_names_for_fold) && length(cat_names_for_fold) > 0) {
          for (cat_col_name in cat_names_for_fold) {
            if (cat_col_name %in% colnames(X_train_fold_current)) {
              X_train_fold_current[[cat_col_name]] <- as.factor(as.character(X_train_fold_current[[cat_col_name]]))
            }
            if (cat_col_name %in% colnames(X_test_fold_current)) {
              if (cat_col_name %in% colnames(X_train_fold_current) && is.factor(X_train_fold_current[[cat_col_name]])) {
                train_levels <- levels(X_train_fold_current[[cat_col_name]])
                X_test_fold_current[[cat_col_name]] <- factor(as.character(X_test_fold_current[[cat_col_name]]), levels = train_levels)
              } else {
                X_test_fold_current[[cat_col_name]] <- as.factor(as.character(X_test_fold_current[[cat_col_name]]))
              }
            }
          }
        }
        
        # --- Separate Target Variable ---
        # Assuming TARGET_COLUMN_NAME is globally defined and present in the dataframes
        if (!TARGET_COLUMN_NAME %in% colnames(X_train_fold_current) || !TARGET_COLUMN_NAME %in% colnames(X_test_fold_current)) {
          cat(paste0("    ERROR: Target column '", TARGET_COLUMN_NAME, "' not found for fold ", fold_number_from_json, ". Skipping this fold.\n"))
          next
        }
        
        # --- Store the prepared data for this fold ---
        # Each element of real_dataset_prepared_folds_list will be a list containing the X_train, y_train, etc. for that fold
        real_dataset_prepared_folds_list[[fold_idx_json]] <- list(
          fold_number = fold_number_from_json,
          train_set = X_train_fold_current,
          test_set = X_test_fold_current,
          cat_names = cat_names_for_fold
        )
        cat(paste("    Successfully prepared and stored data for fold", fold_number_from_json, "\n"))
      } # End of loop over folds from JSON
      
      
    } else { # End of if block for checking if all_folds_data_from_json is valid
      cat(paste("  No valid fold data loaded from JSON for '", base_name, "'. This dataset will be skipped for modeling.\n"))
      # Ensure real_dataset_prepared_folds_list is empty or handled appropriately by downstream code
      real_dataset_prepared_folds_list <- list() # Make it an empty list
    }
  }
  
  # D. Loop through models specified in MODELS_TO_RUN
  for (model_name_to_run in MODELS_TO_RUN) {
    cat(paste("    --- Running Model:", model_name_to_run, "---\n"))
    
    start_time <- Sys.time()
    
    if (model_name_to_run == "M5P_YJ") {
      # Call run_m5p_nested_cv_yj ONCE per dataset, providing all pre-defined folds.
      # This function will internally loop through these folds, do YJ, tune M, train, and test.
      m5p_full_output <- NULL
      
      # Two different inputs for the function used, one for "real" dataset and the other for the rest
      if (base_name != "real") {
        tryCatch({
          m5p_full_output <- run_m5p_nested_cv_yj(
            full_data = globally_preprocessed_data,
            target_col_name = actual_target_name,
            outer_fold_indices_list = outer_fold_indices_list_1based, # Give it all folds
            m_vals = c(1, 5, 10, 20), # Your M values for tuning
            inner_k = 5,             # Inner CV folds for M tuning
            random_seed = 123,
            data_set = base_name
          )
        }, error = function(e) {
          cat(paste("    ERROR running M5P_YJ for dataset", base_name, ":", e$message, "\n"))
        })
      } else {
        tryCatch({
          m5p_full_output <- run_m5p_nested_cv_yj(
            full_data_list = real_dataset_prepared_folds_list,
            target_col_name = actual_target_name,
            outer_fold_indices_list = outer_fold_indices_list_1based,
            m_vals = c(1, 5, 10, 20), # Your M values for tuning
            inner_k = 5,             # Inner CV folds for M tuning
            random_seed = 123,
            data_set = base_name
          )
        }, error = function(e) {
          cat(paste("    ERROR running M5P_YJ for dataset", base_name, ":", e$message, "\n"))
        })
      }
      
      end_time <- Sys.time()
      #Calculate the time it took to run the model
      total_time_taken_for_m5p <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      if (!is.null(m5p_full_output) && !is.null(m5p_full_output$results_per_fold) && nrow(m5p_full_output$results_per_fold) > 0) {
        time_per_fold_approx <- total_time_taken_for_m5p / nrow(m5p_full_output$results_per_fold)
        
        # Collect the results
        for (k_fold_m5p in 1:nrow(m5p_full_output$results_per_fold)) {
          fold_result_row <- m5p_full_output$results_per_fold[k_fold_m5p, ]
          
          current_result_df_row <- data.frame(
            Dataset = base_name,
            Fold = fold_result_row$Fold, # This is the fold number from M5P's internal loop
            Model = model_name_to_run,
            MSE = ifelse(!is.na(fold_result_row$M5P_MSE), fold_result_row$M5P_MSE, NA),
            Time = time_per_fold_approx, # Approximate time for this specific fold
            stringsAsFactors = FALSE
          )
          all_results_data[[length(all_results_data) + 1]] <- current_result_df_row
          cat(paste("      M5P_YJ Fold", fold_result_row$Fold, "MSE:", round(fold_result_row$M5P_MSE, 4), 
                    "(Best M:", fold_result_row$Best_M, ")\n"))
        }
        cat(paste("    M5P_YJ Mean MSE:", round(m5p_full_output$mean_mse, 4), 
                  "Total Time:", round(total_time_taken_for_m5p, 2), "s\n"))
      } else {
        # Log a single failure entry if the whole M5P run failed for the dataset
        current_result_df_row <- data.frame(
          Dataset = base_name, Fold = NA, Model = model_name_to_run,
          MSE = NA, Time = total_time_taken_for_m5p, stringsAsFactors = FALSE
        )
        all_results_data[[length(all_results_data) + 1]] <- current_result_df_row
      }
    } else {
      cat(paste("    WARNING: Model '", model_name_to_run, "' logic not defined. Skipping.\n"))
      end_time <- Sys.time() # Still record time for consistency if block was entered
      time_taken_unknown <- as.numeric(difftime(end_time, start_time, units = "secs"))
      current_result_df_row <- data.frame(
        Dataset = base_name, Fold = NA, Model = model_name_to_run,
        MSE = NA, Time = time_taken_unknown, stringsAsFactors = FALSE)
      all_results_data[[length(all_results_data) + 1]] <- current_result_df_row
    }
  } # End model loop
} # End dataset loop

# Combine all results and write to CSV
if (length(all_results_data) > 0) {
  final_benchmark_results_df <- do.call(rbind, all_results_data)
  cat("\n\n===== Benchmark Run Complete. Summary of Results: =====\n")
  # Sort for easier viewing
  final_benchmark_results_df <- final_benchmark_results_df[order(final_benchmark_results_df$Dataset, 
                                                                 final_benchmark_results_df$Model, 
                                                                 final_benchmark_results_df$Fold), ]
  print(final_benchmark_results_df)
  if (!exists("OUTPUT_RESULTS_FILE") || is.na(OUTPUT_RESULTS_FILE) || OUTPUT_RESULTS_FILE == "") {
    cat("\nWARNING: OUTPUT_RESULTS_FILE not defined. Results not saved.\n")
  } else {
    tryCatch({
      write.csv(final_benchmark_results_df, OUTPUT_RESULTS_FILE, row.names = FALSE)
      cat(paste("\nBenchmark results successfully written to:", OUTPUT_RESULTS_FILE, "\n"))
    }, error = function(e) {
      cat(paste("\nERROR writing results to CSV '", OUTPUT_RESULTS_FILE, "': ", e$message, "\n"))
    })
  }
} else {
  cat("\nNo results were generated.\n")
}


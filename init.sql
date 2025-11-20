-- Initialize MySQL database with proper settings
SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";

-- Create database if not exists (should already exist from env)
CREATE DATABASE IF NOT EXISTS data_platform CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE data_platform;

-- Set timezone
SET time_zone = '+00:00';

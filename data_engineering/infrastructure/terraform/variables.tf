# Terraform variables for Board Game NLP infrastructure

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "cluster_version" {
  description = "Kubernetes cluster version"
  type        = string
  default     = "1.28"
}

variable "node_instance_types" {
  description = "EC2 instance types for EKS nodes"
  type        = list(string)
  default     = ["t3.medium"]
}

variable "ml_node_instance_types" {
  description = "EC2 instance types for ML workloads"
  type        = list(string)
  default     = ["g4dn.xlarge"]
}

variable "min_nodes" {
  description = "Minimum number of nodes"
  type        = number
  default     = 1
}

variable "max_nodes" {
  description = "Maximum number of nodes"
  type        = number
  default     = 10
}

variable "desired_nodes" {
  description = "Desired number of nodes"
  type        = number
  default     = 3
}

variable "ml_min_nodes" {
  description = "Minimum number of ML nodes"
  type        = number
  default     = 0
}

variable "ml_max_nodes" {
  description = "Maximum number of ML nodes"
  type        = number
  default     = 5
}

variable "ml_desired_nodes" {
  description = "Desired number of ML nodes"
  type        = number
  default     = 0
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 100
}

variable "db_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  type        = number
  default     = 1000
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_clusters" {
  description = "Number of Redis cache clusters"
  type        = number
  default     = 2
}

variable "enable_ml_nodes" {
  description = "Enable ML nodes with GPU support"
  type        = bool
  default     = false
}

variable "enable_monitoring" {
  description = "Enable detailed monitoring"
  type        = bool
  default     = true
}

variable "backup_retention_period" {
  description = "RDS backup retention period in days"
  type        = number
  default     = 7
}

variable "enable_encryption" {
  description = "Enable encryption for all resources"
  type        = bool
  default     = true
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the cluster"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "ACM certificate ARN for HTTPS"
  type        = string
  default     = ""
}

variable "enable_autoscaling" {
  description = "Enable cluster autoscaling"
  type        = bool
  default     = true
}

variable "autoscaling_min_size" {
  description = "Minimum size for cluster autoscaling"
  type        = number
  default     = 1
}

variable "autoscaling_max_size" {
  description = "Maximum size for cluster autoscaling"
  type        = number
  default     = 20
}

variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_types" {
  description = "Instance types for spot instances"
  type        = list(string)
  default     = ["t3.medium", "t3.large", "t3.xlarge"]
}

variable "enable_gpu_instances" {
  description = "Enable GPU instances for ML workloads"
  type        = bool
  default     = false
}

variable "gpu_instance_types" {
  description = "GPU instance types for ML workloads"
  type        = list(string)
  default     = ["g4dn.xlarge", "g4dn.2xlarge", "p3.2xlarge"]
}

variable "data_retention_days" {
  description = "Data retention period in days"
  type        = number
  default     = 90
}

variable "enable_data_lake_lifecycle" {
  description = "Enable S3 lifecycle policies for data lake"
  type        = bool
  default     = true
}

variable "lifecycle_transition_days" {
  description = "Days before transitioning to IA storage"
  type        = number
  default     = 30
}

variable "lifecycle_glacier_days" {
  description = "Days before transitioning to Glacier storage"
  type        = number
  default     = 90
}

variable "enable_vpc_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = false
}

variable "flow_log_retention_days" {
  description = "VPC flow log retention period in days"
  type        = number
  default     = 30
}

variable "enable_cloudtrail" {
  description = "Enable AWS CloudTrail"
  type        = bool
  default     = false
}

variable "cloudtrail_retention_days" {
  description = "CloudTrail log retention period in days"
  type        = number
  default     = 90
}

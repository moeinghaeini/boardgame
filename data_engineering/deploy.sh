#!/bin/bash

# Board Game NLP - Data Engineering Deployment Script
# This script deploys the complete data engineering solution

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${ENVIRONMENT:-dev}
NAMESPACE=${NAMESPACE:-boardgame-nlp}
REGION=${AWS_REGION:-us-west-2}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    local tools=("docker" "kubectl" "helm" "terraform" "aws")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    # Check Kubernetes cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Kubernetes cluster not accessible"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

build_images() {
    log_info "Building Docker images..."
    
    # Build API image
    log_info "Building API image..."
    docker build -t boardgame-nlp-api:latest -f api/Dockerfile .
    
    # Build ML pipeline image
    log_info "Building ML pipeline image..."
    docker build -t boardgame-nlp-ml:latest -f pipelines/Dockerfile .
    
    # Build data pipeline image
    log_info "Building data pipeline image..."
    docker build -t boardgame-nlp-data:latest -f pipelines/Dockerfile .
    
    log_success "Docker images built successfully"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure with Terraform..."
    
    cd infrastructure/terraform
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -var="environment=$ENVIRONMENT" -var="aws_region=$REGION"
    
    # Apply infrastructure
    terraform apply -var="environment=$ENVIRONMENT" -var="aws_region=$REGION" -auto-approve
    
    cd ../..
    
    log_success "Infrastructure deployed successfully"
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    kubectl apply -f infrastructure/kubernetes/namespace.yaml
    kubectl apply -f infrastructure/kubernetes/api-deployment.yaml
    kubectl apply -f infrastructure/kubernetes/airflow-deployment.yaml
    kubectl apply -f infrastructure/kubernetes/monitoring-deployment.yaml
    
    log_success "Kubernetes resources deployed successfully"
}

deploy_with_helm() {
    log_info "Deploying with Helm..."
    
    # Add Helm repositories
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add apache-airflow https://airflow.apache.org
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Deploy PostgreSQL
    helm upgrade --install postgresql bitnami/postgresql \
        --namespace $NAMESPACE \
        --set auth.postgresPassword=boardgame_nlp_password \
        --set auth.database=boardgame_nlp \
        --set primary.persistence.size=20Gi
    
    # Deploy Redis
    helm upgrade --install redis bitnami/redis \
        --namespace $NAMESPACE \
        --set auth.enabled=false \
        --set master.persistence.size=10Gi
    
    # Deploy Airflow
    helm upgrade --install airflow apache-airflow/airflow \
        --namespace $NAMESPACE \
        --set executor=LocalExecutor \
        --set postgresql.enabled=false \
        --set externalDatabase.host=postgresql \
        --set externalDatabase.database=boardgame_nlp \
        --set externalDatabase.user=postgres \
        --set externalDatabase.password=boardgame_nlp_password \
        --set redis.enabled=false \
        --set externalRedis.host=redis-master \
        --set externalRedis.port=6379
    
    # Deploy Prometheus
    helm upgrade --install prometheus prometheus-community/prometheus \
        --namespace $NAMESPACE \
        --set server.persistentVolume.size=20Gi \
        --set alertmanager.persistentVolume.size=10Gi
    
    # Deploy Grafana
    helm upgrade --install grafana grafana/grafana \
        --namespace $NAMESPACE \
        --set persistence.size=10Gi \
        --set adminPassword=admin \
        --set service.type=LoadBalancer
    
    log_success "Helm deployments completed successfully"
}

deploy_data_pipelines() {
    log_info "Deploying data pipelines..."
    
    # Copy DAGs to Airflow
    kubectl cp dags/ $NAMESPACE/airflow-webserver-0:/opt/airflow/dags/
    kubectl cp pipelines/ $NAMESPACE/airflow-webserver-0:/opt/airflow/pipelines/
    
    # Restart Airflow to load new DAGs
    kubectl rollout restart deployment/airflow-webserver -n $NAMESPACE
    kubectl rollout restart deployment/airflow-scheduler -n $NAMESPACE
    
    log_success "Data pipelines deployed successfully"
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Apply monitoring configurations
    kubectl apply -f monitoring/prometheus/
    kubectl apply -f monitoring/grafana/
    
    # Wait for services to be ready
    kubectl wait --for=condition=ready pod -l app=prometheus -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=grafana -n $NAMESPACE --timeout=300s
    
    log_success "Monitoring setup completed"
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    kubectl get pods -n $NAMESPACE
    
    # Check services
    kubectl get services -n $NAMESPACE
    
    # Check ingress
    kubectl get ingress -n $NAMESPACE
    
    # Test API health
    local api_url=$(kubectl get service boardgame-nlp-api-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if [ -n "$api_url" ]; then
        curl -f "http://$api_url:8000/health" || log_warning "API health check failed"
    fi
    
    log_success "Deployment verification completed"
}

show_access_info() {
    log_info "Deployment completed! Access information:"
    
    echo ""
    echo "=== Service URLs ==="
    
    # API
    local api_url=$(kubectl get service boardgame-nlp-api-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if [ -n "$api_url" ]; then
        echo "API: http://$api_url:8000"
        echo "API Docs: http://$api_url:8000/docs"
    fi
    
    # Airflow
    local airflow_url=$(kubectl get service airflow-webserver -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if [ -n "$airflow_url" ]; then
        echo "Airflow: http://$airflow_url:8080"
    fi
    
    # Grafana
    local grafana_url=$(kubectl get service grafana -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if [ -n "$grafana_url" ]; then
        echo "Grafana: http://$grafana_url:3000 (admin/admin)"
    fi
    
    # Prometheus
    local prometheus_url=$(kubectl get service prometheus-server -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if [ -n "$prometheus_url" ]; then
        echo "Prometheus: http://$prometheus_url:9090"
    fi
    
    echo ""
    echo "=== Useful Commands ==="
    echo "kubectl get pods -n $NAMESPACE"
    echo "kubectl logs -f deployment/boardgame-nlp-api -n $NAMESPACE"
    echo "kubectl port-forward service/boardgame-nlp-api-service 8000:8000 -n $NAMESPACE"
    echo ""
    echo "=== Next Steps ==="
    echo "1. Configure your domain and SSL certificates"
    echo "2. Set up monitoring alerts"
    echo "3. Configure data sources in Grafana"
    echo "4. Run initial data collection pipeline"
    echo "5. Train your first ML model"
}

# Main deployment function
main() {
    log_info "Starting Board Game NLP Data Engineering deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Region: $REGION"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-infrastructure)
                SKIP_INFRASTRUCTURE=true
                shift
                ;;
            --skip-kubernetes)
                SKIP_KUBERNETES=true
                shift
                ;;
            --skip-helm)
                SKIP_HELM=true
                shift
                ;;
            --skip-monitoring)
                SKIP_MONITORING=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --skip-infrastructure  Skip infrastructure deployment"
                echo "  --skip-kubernetes     Skip Kubernetes deployment"
                echo "  --skip-helm          Skip Helm deployment"
                echo "  --skip-monitoring    Skip monitoring setup"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run deployment steps
    check_prerequisites
    
    if [ "$SKIP_INFRASTRUCTURE" != "true" ]; then
        deploy_infrastructure
    fi
    
    if [ "$SKIP_KUBERNETES" != "true" ]; then
        deploy_kubernetes
    fi
    
    if [ "$SKIP_HELM" != "true" ]; then
        deploy_with_helm
    fi
    
    deploy_data_pipelines
    
    if [ "$SKIP_MONITORING" != "true" ]; then
        setup_monitoring
    fi
    
    verify_deployment
    show_access_info
    
    log_success "Board Game NLP Data Engineering deployment completed successfully!"
}

# Run main function
main "$@"

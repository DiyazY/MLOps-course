# MLOps Laboratory Assignment: Learning Kubernetes

**Course:** ITC.CEE.310-04-2025-2026-1 Machine Learning and Operations – MLOps
**Duration:** 2-3 hours
**Prerequisites:** Basic Linux command line knowledge

---

## Learning Objectives

By completing this lab, you will:

- Set up and navigate a local Kubernetes cluster
- Deploy a multi-service ML architecture (API Gateway + ML Inference)
- Configure inter-service communication using Kubernetes DNS
- Manage configuration with ConfigMaps and Secrets
- Implement autoscaling and rolling updates

---

## Scenario

You are an MLOps engineer deploying a **production ML inference system** with two services:

1. **API Gateway** (nginx) - Receives client requests, routes them to the ML service, handles load balancing
2. **ML Inference Service** (http-echo) - Simulates a model serving endpoint returning predictions

```
┌─────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Client    │ ──────► │   API Gateway   │ ──────► │   ML Service    │
│  (curl)     │         │    (nginx)      │         │  (http-echo)    │
└─────────────┘         └─────────────────┘         └─────────────────┘
                         Port: 30080                  Internal only
                         (NodePort)                   (ClusterIP)
```

**Pre-built images used:**

- `nginx:alpine` - Lightweight API gateway
- `hashicorp/http-echo` - Simple HTTP echo service (simulates ML inference endpoint)

---

## Pre-Lab Setup

Install before starting:

- Docker Desktop (or Docker Engine on Linux)
- Minikube (v1.30+)
- kubectl CLI
- k9s (optional, for cluster navigation)

---

## [A Cheatsheet on Kubernetes](https://blog.bytebytego.com/p/kubernetes-made-easy-a-beginners)

![Kubernetes Cheatsheet](https://substackcdn.com/image/fetch/$s_!Oyvm!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4876c777-96bb-4563-bc98-60561dbc4442_1493x1600.png)

- For deeper understanding, refer to the [Kubernetes Documentation](https://kubernetes.io/docs/home/).

---

# PART 1: FOUNDATION (Easy)

## Step 1: Setting Up Your Kubernetes Cluster

**Difficulty:** Easy | **Time:** 10 minutes

### Tasks

**1.1** Start a Minikube cluster:

```bash
minikube start --cpus=2 --memory=4096
```

**1.2** Verify the cluster is running:

```bash
kubectl cluster-info
kubectl get nodes
```

**1.3** Create a namespace for this lab:

```bash
kubectl create namespace mlops-lab
kubectl config set-context --current --namespace=mlops-lab
```

### Questions

> **Q1.1:** What does `kubectl cluster-info` show you? What is the role of the Kubernetes control plane?

> **Q1.2:** Why do we create a separate namespace instead of using `default`? Name two benefits of using namespaces.

---

## Step 2: Deploying the ML Inference Service

**Difficulty:** Easy | **Time:** 15 minutes

We'll deploy the ML inference service first - this is the backend that simulates an ML model endpoint. We use `hashicorp/http-echo` as a lightweight service that returns a JSON response (simulating model predictions).

### Tasks

**2.1** Create `ml-service-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
  namespace: mlops-lab
  labels:
    app: ml-inference
    tier: backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-inference
  template:
    metadata:
      labels:
        app: ml-inference
        tier: backend
    spec:
      containers:
        - name: ml-model
          image: hashicorp/http-echo:latest
          args:
            - "-listen=:5000"
            - '-text={"predictions": [2.5, 3.0, 4.5], "model": "half_plus_two", "version": "1.0"}'
          ports:
            - containerPort: 5000
              name: http
          resources:
            requests:
              memory: "64Mi"
              cpu: "50m"
            limits:
              memory: "128Mi"
              cpu: "100m"
          readinessProbe:
            httpGet:
              path: /
              port: 5000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /
              port: 5000
            initialDelaySeconds: 10
            periodSeconds: 15
```

> **Note:** In production, this would be a real ML serving framework like TensorFlow Serving, TorchServe, or Triton. We use http-echo for simplicity - it returns a fixed JSON response simulating a model prediction.

**2.2** Create `ml-service-service.yaml` (ClusterIP - internal only):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-inference-service
  namespace: mlops-lab
  labels:
    app: ml-inference
spec:
  type: ClusterIP
  selector:
    app: ml-inference
  ports:
    - name: http
      port: 5000
      targetPort: 5000
```

**2.3** Apply both resources:

```bash
kubectl apply -f ml-service-deployment.yaml
kubectl apply -f ml-service-service.yaml
kubectl rollout status deployment/ml-inference
```

**2.4** Verify the service is running:

```bash
kubectl get pods -l app=ml-inference
kubectl get services
```

### Questions

> **Q2.1:** Why do we use `ClusterIP` instead of `NodePort` for the ML service? What is the security benefit?

> **Q2.2:** What is the full DNS name that other pods can use to reach this service? (Hint: `<service>.<namespace>.svc.cluster.local`)

> **Q2.3:** We're using a mock service (http-echo) instead of real TensorFlow Serving. What would be different if we used a real ML serving framework?

---

## Step 3: Deploying the API Gateway

**Difficulty:** Easy | **Time:** 20 minutes

Now we'll deploy nginx as an API gateway that routes requests to the ML service.

### Tasks

**3.1** Create `gateway-configmap.yaml` with nginx configuration:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gateway-config
  namespace: mlops-lab
data:
  nginx.conf: |
    events {
        worker_connections 1024;
    }

    http {
        upstream ml_backend {
            server ml-inference-service:5000;
        }

        server {
            listen 80;

            # Health check endpoint for the gateway itself
            location /health {
                return 200 'Gateway OK\n';
                add_header Content-Type text/plain;
            }

            # Route prediction requests to ML service
            location /predict {
                proxy_pass http://ml_backend/;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Request-ID $request_id;
                proxy_connect_timeout 10s;
                proxy_read_timeout 30s;
            }

            # Status page
            location /status {
                return 200 'API Gateway v1.0\nBackend: ml-inference-service:5000\n';
                add_header Content-Type text/plain;
            }
        }
    }
```

**3.2** Create `gateway-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: mlops-lab
  labels:
    app: api-gateway
    tier: frontend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
        tier: frontend
    spec:
      containers:
        - name: nginx
          image: nginx:alpine
          ports:
            - containerPort: 80
              name: http
          volumeMounts:
            - name: nginx-config
              mountPath: /etc/nginx/nginx.conf
              subPath: nginx.conf
              readOnly: true
          resources:
            requests:
              memory: "64Mi"
              cpu: "100m"
            limits:
              memory: "128Mi"
              cpu: "200m"
          readinessProbe:
            httpGet:
              path: /health
              port: 80
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 80
            initialDelaySeconds: 10
            periodSeconds: 15
      volumes:
        - name: nginx-config
          configMap:
            name: gateway-config
```

**3.3** Create `gateway-service.yaml` (NodePort - external access):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-service
  namespace: mlops-lab
  labels:
    app: api-gateway
spec:
  type: NodePort
  selector:
    app: api-gateway
  ports:
    - name: http
      port: 80
      targetPort: 80
      nodePort: 30080
```

**3.4** Apply all gateway resources:

```bash
kubectl apply -f gateway-configmap.yaml
kubectl apply -f gateway-deployment.yaml
kubectl apply -f gateway-service.yaml
kubectl rollout status deployment/api-gateway
```

### Questions

> **Q3.1:** In the nginx config, we use `server ml-inference-service:5000`. How does nginx resolve this hostname inside the cluster?

> **Q3.2:** Why do we mount the ConfigMap as a file instead of using environment variables for nginx configuration?

---

## Step 4: Testing Service-to-Service Communication

**Difficulty:** Easy | **Time:** 15 minutes

### Tasks

**4.1** Get the gateway URL:

**Option A - Using minikube service (Recommended):**

```bash
# This opens a tunnel and returns the correct URL
minikube service api-gateway-service -n mlops-lab --url
```

Copy the URL it returns and set it:

```bash
export GATEWAY_URL=
```

**Option B - Using minikube tunnel (for LoadBalancer-like access):**

```bash
# In a separate terminal, run:
minikube tunnel

# Then in your main terminal:
export GATEWAY_URL=http://$(minikube ip):30080
```

**Option C - Using port-forward (most reliable):**

```bash
# In a separate terminal:
kubectl port-forward svc/api-gateway-service 8080:80 -n mlops-lab

# Then in your main terminal:
export GATEWAY_URL=http://localhost:8080
```

Verify the URL works:

```bash
echo "Gateway URL: $GATEWAY_URL"
```

**4.2** Test the gateway health endpoint:

```bash
curl $GATEWAY_URL/health
```

**4.3** Test the gateway status:

```bash
curl $GATEWAY_URL/status
```

**4.4** Send a prediction request through the gateway to the ML service:

```bash
curl $GATEWAY_URL/predict
```

Expected output: `{"predictions": [2.5, 3.0, 4.5], "model": "half_plus_two", "version": "1.0"}`

**4.5** Verify the request flow by checking logs:

```bash
# Check gateway logs
kubectl logs -l app=api-gateway --tail=10

# Check ML service logs
kubectl logs -l app=ml-inference --tail=10
```

Notes:

- You may need to check logs from multiple pods if there are multiple replicas.
- You may increase `--tail` to see more logs.

### Questions

> **Q4.1:** Trace the complete request path from your curl command to the ML model response. List each component the request passes through.

> **Q4.2:** If the ML service has 2 replicas, which pod handles the request? How is this decided?

---

## Step 5: Understanding Service Discovery

**Difficulty:** Easy | **Time:** 15 minutes

### Tasks

**5.1** Create a debug pod to explore DNS (note: we run it in the same namespace):

```bash
kubectl run debug-pod --image=busybox:1.36 --rm -it --restart=Never -n mlops-lab -- /bin/sh
```

**5.2** Inside the debug pod, test DNS resolution:

```sh
# Look up the ML service
nslookup ml-inference-service

# Look up with full domain
nslookup ml-inference-service.mlops-lab.svc.cluster.local

# Look up the gateway service
nslookup api-gateway-service

# Test connectivity to ML service
wget -q -O- http://ml-inference-service:5000/

# Exit the debug pod
exit
```

**5.3** View the DNS configuration:

```bash
kubectl run dns-check --image=busybox:1.36 --rm -it --restart=Never -n mlops-lab -- cat /etc/resolv.conf
```

### Questions

> **Q5.1:** What IP address does `ml-inference-service` resolve to? Is this a pod IP or something else?

> **Q5.2:** The `/etc/resolv.conf` shows a `search` directive. What does this do and why is it useful?
>
> **Hint:** If you run the debug pod in a different namespace (e.g., `default`), `nslookup ml-inference-service` will fail until it tries the full name. This demonstrates how the search path works!

> **Q5.3:** If you scale the ML service to 5 replicas, does the DNS record change? Explain how Kubernetes handles this.

---

# PART 2: INTEGRATION (Medium)

## Step 6: Advanced ConfigMap Usage

**Difficulty:** Medium | **Time:** 15 minutes

### Tasks

**6.1** Create a shared ConfigMap for both services - `shared-config.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: shared-config
  namespace: mlops-lab
data:
  ENVIRONMENT: "development"
  LOG_LEVEL: "info"
  MODEL_NAME: "half_plus_two"
  REQUEST_TIMEOUT: "30"
```

**6.2** Create an ML-specific ConfigMap - `ml-config.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
  namespace: mlops-lab
data:
  TENSORFLOW_INTRA_OP_PARALLELISM: "2"
  TENSORFLOW_INTER_OP_PARALLELISM: "2"
  BATCH_SIZE: "32"
```

**6.3** Update the ML deployment to use ConfigMaps. Modify `ml-service-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
  namespace: mlops-lab
  labels:
    app: ml-inference
    tier: backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-inference
  template:
    metadata:
      labels:
        app: ml-inference
        tier: backend
    spec:
      containers:
        - name: ml-model
          image: hashicorp/http-echo:latest
          args:
            - "-listen=:5000"
            - '-text={"predictions": [2.5, 3.0, 4.5], "model": "half_plus_two", "version": "1.0"}'
          ports:
            - containerPort: 5000
              name: http
          envFrom:
            - configMapRef:
                name: shared-config
            - configMapRef:
                name: ml-config
          resources:
            requests:
              memory: "64Mi"
              cpu: "50m"
            limits:
              memory: "128Mi"
              cpu: "100m"
          readinessProbe:
            httpGet:
              path: /
              port: 5000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /
              port: 5000
            initialDelaySeconds: 10
            periodSeconds: 15
```

**6.4** Apply and verify:

```bash
kubectl apply -f shared-config.yaml
kubectl apply -f ml-config.yaml
kubectl apply -f ml-service-deployment.yaml
kubectl rollout status deployment/ml-inference
```

**6.5** Verify environment variables are configured (since http-echo is a minimal image without shell utilities, we use kubectl to inspect the pod spec):

```bash
# Method 1: Check the pod spec shows the ConfigMap references
kubectl get pod -l app=ml-inference -o jsonpath='{.items[0].spec.containers[0].envFrom[*].configMapRef.name}'
# Expected output: shared-config ml-config

# Method 2: Describe the pod to see environment configuration
kubectl describe pod -l app=ml-inference | grep -A 5 "Environment Variables from"
```

Expected output for Method 2:

```
Environment Variables from:
  shared-config  ConfigMap  Optional: false
  ml-config      ConfigMap  Optional: false
```

> **Note:** The http-echo image is a minimal "scratch" container without a shell or standard utilities. We can't run `kubectl exec ... -- env` to see the actual environment variables at runtime. In production ML images (like TensorFlow Serving), you would be able to inspect environment variables directly inside the container.
>
> **Note:** Notice the service discovery environment variables that Kubernetes automatically injects (like `ML_INFERENCE_SERVICE_SERVICE_HOST`). These are created for every service in the namespace!

### Questions

> **Q6.1:** We use two ConfigMaps (`shared-config` and `ml-config`). What is the advantage of separating configuration this way?

> **Q6.2:** If both ConfigMaps define the same key, which value would the pod receive? Test this if unsure.

---

## Step 7: Managing Secrets

**Difficulty:** Medium | **Time:** 15 minutes

### Tasks

**7.1** Create secrets for API authentication:

```bash
kubectl create secret generic api-credentials \
  --namespace=mlops-lab \
  --from-literal=API_KEY=mlops-secret-key-2026 \
  --from-literal=JWT_SECRET=super-secret-jwt-token
```

**7.2** Create a secret for ML model access (simulating model registry credentials):

```bash
kubectl create secret generic ml-registry-credentials \
  --namespace=mlops-lab \
  --from-literal=REGISTRY_USER=mlops-user \
  --from-literal=REGISTRY_PASSWORD=registry-password-xyz
```

**7.3** View and decode secrets:

```bash
# List secrets
kubectl get secrets

# View secret details
kubectl describe secret api-credentials

# Decode a value
kubectl get secret api-credentials -o jsonpath='{.data.API_KEY}' | base64 --decode
echo ""
```

**7.4** Update gateway deployment to use secrets. Modify `gateway-deployment.yaml` container spec:

```yaml
env:
  - name: API_KEY
    valueFrom:
      secretKeyRef:
        name: api-credentials
        key: API_KEY
```

Apply the change:

```bash
kubectl apply -f gateway-deployment.yaml
kubectl rollout status deployment/api-gateway
```

### Questions

> **Q7.1:** Why are Secrets base64 encoded instead of encrypted by default? What does this mean for security?

> **Q7.2:** List three types of sensitive data an ML inference system might need to store as Secrets.

> **Q7.3:** What Kubernetes feature would you enable to encrypt Secrets at rest?

---

## Step 8: Scaling and Load Balancing

**Difficulty:** Medium | **Time:** 20 minutes

### Tasks

**8.1** Scale the ML service manually:

```bash
kubectl scale deployment/ml-inference --replicas=4
kubectl get pods -l app=ml-inference -w
```

**8.2** Test load balancing by sending multiple requests:

```bash
# Send 10 requests and observe distribution
for i in {1..10}; do
  echo "Request $i:"
  curl -s $GATEWAY_URL/predict
  echo ""
done
```

**8.3** Watch which pods handle requests:

```bash
# In one terminal, watch pod logs
kubectl logs -f -l app=ml-inference --prefix=true

# In another terminal, send requests
curl $GATEWAY_URL/predict
```

**8.4** Test resilience - delete a pod during requests:

```bash
# Terminal 1: Continuous requests
while true; do
  curl -s -o /dev/null -w "Status: %{http_code}\n" $GATEWAY_URL/predict
  sleep 0.5
done

# Terminal 2: Delete a pod
kubectl delete pod $(kubectl get pod -l app=ml-inference -o jsonpath='{.items[0].metadata.name}')
```

**8.5** Scale back down:

```bash
kubectl scale deployment/ml-inference --replicas=2
```

### Questions

> **Q8.1:** When you deleted a pod during continuous requests, did any requests fail? Why or why not?

> **Q8.2:** The Service uses round-robin load balancing by default. What other load balancing strategies might be better for ML inference? Why?

> **Q8.3:** If one ML pod is slower than others (e.g., running on a node with less CPU), what problems might occur with round-robin?

---

# PART 3: PRODUCTION PATTERNS (Difficult)

## Step 9: Horizontal Pod Autoscaling

**Difficulty:** Difficult | **Time:** 25 minutes

> **Note:** The http-echo container is very lightweight and won't generate enough CPU load from normal requests. To demonstrate HPA scaling, we'll deploy a CPU-intensive service and then also show how to manually simulate scaling behavior.

### Tasks

**9.1** Enable metrics server:

```bash
minikube addons enable metrics-server

# Wait for metrics to be available (1-2 minutes)
kubectl top pods -n mlops-lab
```

**9.2** Deploy a PHP Apache server for load testing (this is the standard K8s HPA demo):

```bash
# Deploy the php-apache server (official K8s HPA example)
kubectl apply -f https://k8s.io/examples/application/php-apache.yaml -n mlops-lab
```

Or create `php-apache.yaml` manually:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: php-apache
  namespace: mlops-lab
spec:
  replicas: 1
  selector:
    matchLabels:
      run: php-apache
  template:
    metadata:
      labels:
        run: php-apache
    spec:
      containers:
        - name: php-apache
          image: registry.k8s.io/hpa-example
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "200m"
            limits:
              cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: php-apache
  namespace: mlops-lab
spec:
  ports:
    - port: 80
  selector:
    run: php-apache
```

> **Note:** This image performs CPU-intensive calculations on each request, making it ideal for demonstrating HPA.

**9.3** Create HPA for php-apache - `php-apache-hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: php-apache-hpa
  namespace: mlops-lab
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: php-apache
  minReplicas: 1
  maxReplicas: 5
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 50
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 30
    scaleUp:
      stabilizationWindowSeconds: 0
```

```bash
kubectl apply -f php-apache-hpa.yaml
```

**9.4** Watch HPA and generate load:

Terminal 1 - Watch HPA:

```bash
watch -n 2 'kubectl get hpa php-apache-hpa -n mlops-lab; echo "---"; kubectl top pods -l run=php-apache -n mlops-lab 2>/dev/null; echo "---"; kubectl get pods -l run=php-apache -n mlops-lab'
```

Terminal 2 - Generate load (scale UP):

```bash
kubectl run -i --tty load-generator --rm --image=busybox:1.36 --restart=Never -n mlops-lab -- /bin/sh -c "while sleep 0.01; do wget -q -O- http://php-apache; done"
```

**Expected behavior:**

1. **With load running:** CPU spikes above 50%, HPA scales up (1 → 2 → 3+ pods)
2. **Stop load (Ctrl+C):** CPU drops, after 30s stabilization window, HPA scales back down to 1

> **Tip:** The scale-up happens quickly (within 1-2 minutes). Scale-down takes longer due to the stabilization window.

**9.5** Now create HPAs for our actual services (for production readiness):

Create `ml-hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
  namespace: mlops-lab
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 50
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 25
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
```

Create `gateway-hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
  namespace: mlops-lab
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 2
  maxReplicas: 6
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
```

```bash
kubectl apply -f ml-hpa.yaml
kubectl apply -f gateway-hpa.yaml
kubectl get hpa
```

**9.6** Clean up the HPA demo:

```bash
kubectl delete deployment php-apache
kubectl delete service php-apache
kubectl delete hpa php-apache-hpa
```

### Questions

> **Q9.1:** The ML HPA has `stabilizationWindowSeconds: 60` for scale-down but `0` for scale-up. Explain why this asymmetry is important for ML services.

> **Q9.2:** We set different CPU thresholds: 50% for ML service, 60% for gateway. Why might the ML service need a lower threshold?

> **Q9.3:** Calculate: If the ML service has 2 pods at 80% CPU utilization and the target is 50%, how many pods will the HPA create?

> **Q9.4:** What are limitations of CPU-based autoscaling for ML inference? Propose a custom metric that might work better.

---

## Step 10: Rolling Updates and Rollbacks

**Difficulty:** Difficult | **Time:** 25 minutes

### Tasks

**10.1** Update deployments with production rolling update strategy.

Modify `ml-service-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
  namespace: mlops-lab
  labels:
    app: ml-inference
    tier: backend
  annotations:
    kubernetes.io/change-cause: "Initial deployment - http-echo v1"
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ml-inference
  template:
    metadata:
      labels:
        app: ml-inference
        tier: backend
        version: v1
    spec:
      terminationGracePeriodSeconds: 30
      containers:
        - name: ml-model
          image: hashicorp/http-echo:latest
          args:
            - "-listen=:5000"
            - '-text={"predictions": [2.5, 3.0, 4.5], "model": "half_plus_two", "version": "1.0"}'
          ports:
            - containerPort: 5000
              name: http
          envFrom:
            - configMapRef:
                name: shared-config
            - configMapRef:
                name: ml-config
          resources:
            requests:
              memory: "64Mi"
              cpu: "50m"
            limits:
              memory: "128Mi"
              cpu: "100m"
          readinessProbe:
            httpGet:
              path: /
              port: 5000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /
              port: 5000
            initialDelaySeconds: 10
            periodSeconds: 15
          lifecycle:
            preStop:
              exec:
                command: ["/bin/sh", "-c", "sleep 10"]
```

**10.2** Apply and verify:

```bash
kubectl apply -f ml-service-deployment.yaml
kubectl rollout status deployment/ml-inference
kubectl rollout history deployment/ml-inference
```

**10.3** Test zero-downtime deployment:

Terminal 1 - Continuous requests:

```bash
while true; do
  RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $GATEWAY_URL/predict)
  echo "$(date +%H:%M:%S) - Status: $RESPONSE"
  sleep 0.3
done
```

Terminal 2 - Trigger update (change the response text to simulate model v2):

```bash
kubectl set image deployment/ml-inference ml-model=hashicorp/http-echo:0.2.3
kubectl annotate deployment/ml-inference kubernetes.io/change-cause="Updated to http-echo 0.2.3 (model v2)" --overwrite

# Watch the rollout
kubectl rollout status deployment/ml-inference
```

Terminal 3 - Watch pods:

```bash
kubectl get pods -l app=ml-inference -w
```

**10.4** View rollout history:

```bash
kubectl rollout history deployment/ml-inference
kubectl rollout history deployment/ml-inference --revision=2
```

**10.5** Practice rollback:

```bash
# Rollback to previous version
kubectl rollout undo deployment/ml-inference

# Watch rollback
kubectl rollout status deployment/ml-inference

# View history after rollback
kubectl rollout history deployment/ml-inference
```

**10.6** Update the gateway simultaneously:

```bash
# Update both services at once
kubectl set image deployment/ml-inference ml-model=hashicorp/http-echo:latest
kubectl set image deployment/api-gateway nginx=nginx:1.25-alpine

# Watch both rollouts
kubectl rollout status deployment/ml-inference &
kubectl rollout status deployment/api-gateway &
wait
```

### Questions

> **Q10.1:** Explain `maxSurge: 1` and `maxUnavailable: 0`. Why is this ideal for ML services requiring high availability?

> **Q10.2:** What is the purpose of the `preStop` lifecycle hook with `sleep 10`? How does it prevent request failures during updates?

> **Q10.3:** During your zero-downtime test, did you see any non-200 status codes? If yes, what might cause these brief failures?

> **Q10.4:** We updated both the gateway and ML service. What's the risk of updating multiple services simultaneously? How would you coordinate updates in production?

> **Q10.5:** Design a canary deployment for this system: How would you route 10% of traffic to a new ML model version while keeping 90% on stable? What Kubernetes resources would you need?

---

## Lab Cleanup

```bash
kubectl delete namespace mlops-lab
minikube stop
```

---

## Architecture Summary

```
                              ┌─────────────────────────────────────────┐
                              │           Kubernetes Cluster            │
                              │                                         │
┌──────────┐  NodePort:30080  │  ┌─────────────────────────────────┐    │
│  Client  │ ───────────────► │  │      API Gateway (nginx)        │    │
│  (curl)  │                  │  │  ┌─────────┐    ┌─────────┐     │    │
└──────────┘                  │  │  │ Pod 1   │    │ Pod 2   │     │    │
                              │  │  └─────────┘    └─────────┘     │    │
                              │  └──────────────┬──────────────────┘    │
                              │                 │                       │
                              │                 │ ClusterIP:5000        │
                              │                 ▼                       │
                              │  ┌─────────────────────────────────┐    │
                              │  │    ML Inference (http-echo)     │    │
                              │  │  ┌─────────┐    ┌─────────┐     │    │
                              │  │  │ Pod 1   │    │ Pod 2   │     │    │
                              │  │  └─────────┘    └─────────┘     │    │
                              │  └─────────────────────────────────┘    │
                              │                                         │
                              └─────────────────────────────────────────┘
```

| Component    | Image               | Service Type | Port       |
| ------------ | ------------------- | ------------ | ---------- |
| API Gateway  | nginx:alpine        | NodePort     | 30080 → 80 |
| ML Inference | hashicorp/http-echo | ClusterIP    | 5000       |

---

## Summary

| Part        | Steps | What You Learned                               |
| ----------- | ----- | ---------------------------------------------- |
| Foundation  | 1-5   | Two-service deployment, Service Discovery, DNS |
| Integration | 6-8   | ConfigMaps, Secrets, Manual Scaling            |
| Production  | 9-10  | HPA Autoscaling, Rolling Updates, Rollbacks    |

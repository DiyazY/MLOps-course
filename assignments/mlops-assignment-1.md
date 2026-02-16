# MLOps Assignment 1: Cloud Platforms

## An Alternative Perspective: DigitalOcean for Budget Conscious MLOps

**Course:** MLOps
**Date:** February 2026

---

## Preface: Why DigitalOcean?

This report examines DigitalOcean rather than AWS, Azure, or GCP. The analysis stems from a real project: I am building a production system and evaluated multiple cloud providers before selecting DigitalOcean based on budget constraints and operational simplicity.

Applying the course framework to a provider outside the "big three" surfaces tradeoffs that hyperscaler only comparisons tend to obscure, particularly around cost predictability and complexity overhead for small teams.

---

## 1) Motivation

**Scenario: Independent Developer Building a Complex System Under Budget Constraints**

This analysis stems from a real architectural decision: building a system complex enough to warrant dedicated server resources, but under strict budget constraints that rule out hyperscaler pricing.

For context: I have production experience with all three major cloud providers (five years with Azure, three years with AWS, and one year with GCP) working on enterprise level systems. This background informs the comparison that follows. The choice of DigitalOcean for my current project is not due to unfamiliarity with hyperscalers, but rather a deliberate decision based on the specific constraints of this project.

The decision matrix looked like this:

| Option                            | Pros                                                                         | Cons                                                                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Dedicated Server (Bare Metal)** | Full control, predictable cost, no noisy neighbors                           | High maintenance burden OS patching, security hardening, backup management, hardware failures become your problem     |
| **Hyperscalers (AWS/Azure/GCP)**  | Managed services reduce operational load, vast ecosystem                     | Cost unpredictability (egress fees, per request pricing), complexity overhead, overkill for single developer projects |
| **DigitalOcean (Middle Ground)**  | Simpler than hyperscalers, more managed than bare metal, transparent pricing | Fewer advanced services, smaller ecosystem                                                                            |

**Why not dedicated servers alone?**

A dedicated server at €50-100/month from providers like Hetzner or OVHcloud offers excellent compute per dollar. However, the maintenance burden is significant:

- Security patches and OS updates are your responsibility
- Backup strategies must be implemented manually
- No managed database failover: if the disk fails at 3 AM, you're the on-call engineer
- Networking complexity when integrating with cloud storage or external services

For a solo developer or small team, this operational overhead directly competes with feature development time.

**Why not AWS/Azure/GCP?**

The hyperscalers solve the maintenance problem but introduce new ones:

- **Cost unpredictability:** A misconfigured S3 bucket or unexpected traffic spike can generate surprise bills. AWS egress fees (~$0.09/GB) accumulate quickly for data-intensive workloads.
- **Complexity tax:** Even simple deployments require understanding IAM policies, VPC configurations, and service quotas. The 200+ service catalog creates decision paralysis.
- **Overkill factor:** Services like SageMaker or Vertex AI are designed for teams with ML platform engineers. For a solo developer running training jobs and inference endpoints, the abstraction layers add friction without proportional benefit.

**Why DigitalOcean?**

DigitalOcean occupies a pragmatic middle ground:

- **Predictable pricing:** Droplets include 1-7TB of bandwidth. No surprise egress charges.
- **Managed services without complexity:** Managed Databases handle backups and failover. Spaces provides S3 compatible storage. These reduce operational burden without requiring IAM expertise.
- **Right-sized for the workload:** The curated service catalog (~30 products) means less time evaluating options, more time building.
- **Escape hatch to dedicated resources:** Dedicated CPU Droplets provide guaranteed compute when needed, while remaining within the same ecosystem.

**The hybrid insight:** DigitalOcean's VPC networking enables a powerful pattern, connect a cost effective dedicated server (for long running compute) to managed cloud services (for storage, databases, CDN) via private networking. This combines the cost efficiency of bare metal with the operational convenience of managed services.

**Assumption:** This analysis assumes workloads that are complex enough to need reliable infrastructure but don't require hyperscaler specific capabilities (TPUs, global scale distributed systems, advanced AutoML).

---

## 2) Platform Decomposition

| Category               | Representative Component                     | Responsibility Boundary                                                                                        | Notes                                                                           |
| ---------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Compute**            | Droplets (VMs) / Dedicated CPU Droplets      | User: application code, runtime configuration. Platform: hypervisor, physical infrastructure, networking stack | Dedicated CPU variants offer guaranteed resources without noisy neighbor issues |
| **Storage**            | Spaces (Object Storage)                      | User: data organization, access patterns. Platform: durability, replication, availability                      | S3 compatible API; 250GB + 1TB transfer for $5/mo                               |
| **Networking**         | VPC (Virtual Private Cloud)                  | User: network topology design, firewall rules. Platform: isolation, routing, private interconnects             | Intra VPC traffic is free; simplifies cost modeling                             |
| **Identity & Access**  | Teams + API Tokens                           | User: team structure, token scoping. Platform: authentication, audit logging                                   | Simpler than IAM policies; tradeoff is less granularity                         |
| **CI/CD & Automation** | App Platform                                 | User: build configuration, deployment triggers. Platform: container orchestration, SSL, scaling                | PaaS layer; abstracts Kubernetes complexity                                     |
| **ML & AI Services**   | GPU Droplets + 1 Click ML Apps               | User: model code, training logic. Platform: GPU drivers, CUDA environment                                      | Limited compared to SageMaker; user owns more of the ML pipeline                |
| **Data & Analytics**   | Managed Databases (PostgreSQL, MySQL, Redis) | User: schema design, query optimization. Platform: backups, failover, patching                                 | Includes automatic daily backups and standby nodes                              |

---

## 3) Representative Components

### Compute: Dedicated CPU Droplets

**What it is:** Virtual machines with guaranteed CPU resources that are not shared with other tenants.

**Responsibility abstracted:** Users avoid the "noisy neighbor" problem common in shared tenancy VMs, where another customer's workload can degrade your performance. The platform guarantees consistent compute capacity without requiring users to understand hypervisor scheduling or CPU pinning.

**Why use it for MLOps:** ML inference workloads are latency sensitive. Shared vCPUs introduce variance that can violate SLAs. Dedicated CPU Droplets provide predictable performance at a fraction of bare metal cost, making them suitable for production inference endpoints in cost constrained environments.

---

### Storage: Spaces Object Storage

**What it is:** S3 compatible object storage with builtin CDN integration.

**Responsibility abstracted:** Users are freed from managing storage infrastructure (disk provisioning, RAID configuration, replication). The platform handles durability (data is replicated across multiple devices) and provides a standard API that works with existing S3 tooling.

**Why use it for MLOps:** ML pipelines generate artifacts, trained models, datasets, logs that need durable, accessible storage. Spaces' S3 compatibility means existing tools (DVC, MLflow artifact stores) work without modification. The included CDN is useful for serving model files to distributed inference nodes.

---

### Networking: VPC (Virtual Private Cloud)

**What it is:** Isolated private networks that span all resources in a datacenter region.

**Responsibility abstracted:** Users define logical network boundaries without configuring physical switches, VLANs, or routing tables. The platform ensures traffic isolation between different customers and handles the underlying SDN (software-defined networking) complexity.

**Why use it for MLOps:** Training jobs often pull large datasets from storage. With VPC, this traffic stays private and, critically, is free. In hyperscalers, inter-service data transfer can accumulate significant costs. VPC simplifies the hybrid pattern: a dedicated server connected via VPN to cloud services communicates without egress fees.

---

### Identity & Access: Teams + API Tokens

**What it is:** Role-based access control for team members and scoped API tokens for programmatic access.

**Responsibility abstracted:** Users don't implement authentication systems or manage credential storage. The platform provides OAuth-based login, token generation, and audit trails.

**Why use it for MLOps:** CI/CD pipelines need credentials to deploy models. Scoped API tokens limit blast radius if compromised, since a token for the staging environment cannot affect production. The simplicity compared to AWS IAM (no policy documents to debug) accelerates onboarding for small teams.

---

### CI/CD & Automation: App Platform

**What it is:** A Platform-as-a-Service (PaaS) layer that builds and deploys applications from Git repositories.

**Responsibility abstracted:** Container orchestration, load balancing, SSL certificate management, and scaling are handled by the platform. Users commit code; the platform handles deployment.

**Why use it for MLOps:** Deploying ML inference APIs typically requires containerization, orchestration, and HTTPS termination. App Platform reduces this to a `Dockerfile` and git push. For startups iterating rapidly on model versions, this removes operational friction while maintaining production grade infrastructure.

---

### ML & AI Services: GPU Droplets

**What it is:** Virtual machines with attached NVIDIA GPUs (typically H100 or older generations) for compute-intensive workloads.

**Responsibility abstracted:** GPU driver installation, CUDA environment configuration, and hardware maintenance are platform responsibilities. Users receive a VM with a working GPU and standard ML frameworks preinstalled.

**Why use it for MLOps:** Training neural networks requires GPU acceleration. While hyperscalers offer managed training services (SageMaker, Vertex AI), these add abstraction and cost that may not suit small scale training. GPU Droplets provide raw GPU access for teams comfortable managing their own training scripts, at transparent hourly rates.

---

### Data & Analytics: Managed Databases

**What it is:** Fully managed relational (PostgreSQL, MySQL) and inmemory (Redis) databases.

**Responsibility abstracted:** The platform handles provisioning, patching, automated backups, failover to standby nodes, and connection pooling. Users interact via standard database protocols.

**Why use it for MLOps:** ML systems need metadata stores (experiment tracking, feature registries) and caching layers (feature stores, prediction caches). Managed databases eliminate the operational burden of database administration, letting ML engineers focus on the pipeline logic rather than `pg_dump` schedules.

---

## 4) Comparison with Other Providers

**Key Observation: A Spectrum, Not a Binary Choice**

The cloud provider landscape isn't a simple "hyperscaler vs. everything else" dichotomy. There's a spectrum based on how much operational responsibility you're willing to trade for cost and simplicity:

```
More Managed / Higher Cost          Less Managed / Lower Cost
        ←―――――――――――――――――――――――――――――――――――――――→

AWS/Azure/GCP    DigitalOcean    OVHcloud/Hetzner    Self hosted
(Full managed)   (Balanced)      (Bare metal focus)  (Full control)
```

**Comparison with AWS/Azure/GCP:**

| Aspect         | Hyperscalers                                                                                          | DigitalOcean                                     |
| -------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| ML training    | Managed services (SageMaker, Vertex AI, Azure ML) with experiment tracking, HPO, distributed training | GPU VMs with raw access; user brings MLflow, W&B |
| Cost model     | Pay-per-use with complex dimensions (compute + storage + egress + API calls)                          | Flat monthly rates with included bandwidth       |
| Hybrid tooling | AWS Outposts, Azure Arc, GCP Anthos, sophisticated but complex                                        | VPC + VPN, simpler but manual setup              |
| Learning curve | Weeks to months for IAM, networking, quotas                                                           | Hours to days for basic deployment               |

**Comparison with OVHcloud (a notable alternative):**

OVHcloud represents the other end of the spectrum, European provider built on dedicated servers (bare metal) with cloud services added later. For hybrid architectures where dedicated compute is primary:

| Aspect            | DigitalOcean                              | OVHcloud                                            |
| ----------------- | ----------------------------------------- | --------------------------------------------------- |
| Core strength     | Developer simplicity, PaaS (App Platform) | Bare metal servers, vRack private networking        |
| Network costs     | Generous included bandwidth (1-7TB)       | **Unlimited** public bandwidth on dedicated servers |
| Hybrid networking | VPC + self managed VPN                    | vRack native private network across all services    |
| Managed services  | Strong (Databases, Spaces, App Platform)  | Available but less mature than DO                   |

**The tradeoff is explicit:** DigitalOcean users accept more operational responsibility than hyperscaler users but less than bare metal users. The platform sits at a pragmatic midpoint, enough managed services to reduce toil, without the complexity tax of 200+ service catalogs.

For budget constrained projects with moderate complexity, this middle ground often provides the best balance of cost, capability, and cognitive load.

---

## 5) Perceived Tradeoffs

### Technical Tradeoff: Limited Specialized ML Infrastructure

**Tradeoff:** DigitalOcean lacks managed ML services comparable to AWS SageMaker, GCP Vertex AI, or Azure Machine Learning.

**Implication:** Users must build or integrate their own:

- Experiment tracking (e.g., self hosted MLflow)
- Model registry (e.g., DVC, custom solutions)
- Feature stores (e.g., Feast, or custom database schemas)
- Distributed training orchestration (e.g., Ray, manual multi node setup)

**When this matters:** Teams training large models, requiring hyperparameter search across many configurations, or needing AutoML capabilities will find DigitalOcean insufficient. The platform suits inference heavy workloads better than training heavy ones.

---

### Organizational Tradeoff: Ecosystem and Enterprise Support Limitations

**Tradeoff:** DigitalOcean has a smaller partner ecosystem, fewer compliance certifications, and less extensive enterprise support compared to hyperscalers.

**Implication:**

- Fewer third party integrations (e.g., Databricks, Snowflake don't offer native DigitalOcean deployments)
- Compliance sensitive industries (healthcare, finance) may find certification gaps (SOC 2 and ISO 27001 are available, but not FedRAMP, HIPAA BAA requires enterprise tier)
- Enterprise support contracts and SLAs are less mature

**When this matters:** Organizations in regulated industries or those requiring extensive vendor ecosystem integration will face friction. Startups targeting enterprise customers may need to migrate platforms as they scale, introducing future technical debt.

---

## 6) Responsibility Boundaries

### User Responsibilities

| Area                      | User Must Handle                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------- |
| **Application Layer**     | Code, dependencies, runtime configuration, model artifacts                              |
| **ML Pipeline**           | Training scripts, data preprocessing, experiment tracking, model versioning             |
| **Security Posture**      | Application level security, secrets management, firewall rule design                    |
| **Data Management**       | Backup strategy beyond platform defaults, data lifecycle policies                       |
| **Scaling Decisions**     | When to add/remove resources; no autoscaling for Droplets                               |
| **Monitoring & Alerting** | Application level metrics, log aggregation (platform provides basic infra metrics only) |

### Platform Responsibilities

| Area                    | Platform Manages                                                |
| ----------------------- | --------------------------------------------------------------- |
| **Infrastructure**      | Physical servers, network hardware, datacenter operations       |
| **Hypervisor Layer**    | VM isolation, resource allocation, live migration               |
| **Managed Services**    | Database backups, failover, patching; object storage durability |
| **Network Fabric**      | VPC isolation, DDoS mitigation, bandwidth provisioning          |
| **Compliance Baseline** | SOC 2, ISO 27001 certification maintenance                      |
| **Platform Security**   | API authentication, control plane security, audit logging       |

### Shared Responsibility Boundary

The boundary sits higher in the stack compared to hyperscalers. DigitalOcean provides less "managed" abstraction, meaning users take on more operational responsibility in exchange for simplicity and cost predictability.

**Visual model:**

```
+---------------------------+
|     User Responsibility   |
|  Application, ML Pipeline,|
|  Monitoring, Scaling      |
+---------------------------+
|   Shared: Network Config, |
|   Access Control          |
+---------------------------+
| Platform Responsibility   |
| Infrastructure, Managed   |
| Services Core Functions   |
+---------------------------+
```

---

## 7) References

1. DigitalOcean. "Droplets Documentation." DigitalOcean Docs, 2026. https://docs.digitalocean.com/products/droplets/

2. DigitalOcean. "Spaces Object Storage." DigitalOcean Docs, 2026. https://docs.digitalocean.com/products/spaces/

3. DigitalOcean. "VPC Networking." DigitalOcean Docs, 2026. https://docs.digitalocean.com/products/networking/vpc/

4. DigitalOcean. "Managed Databases." DigitalOcean Docs, 2026. https://docs.digitalocean.com/products/databases/

5. DigitalOcean. "GPU Droplets." DigitalOcean Docs, 2026. https://docs.digitalocean.com/products/droplets/

6. DigitalOcean. "App Platform." DigitalOcean Docs, 2026. https://docs.digitalocean.com/products/app-platform/

7. DigitalOcean. "Pricing." DigitalOcean, 2026. https://www.digitalocean.com/pricing

8. OVHcloud. "vRack - Private Network." OVHcloud Docs, 2026. https://www.ovhcloud.com/en/network/vrack/

9. OVHcloud. "Dedicated Servers." OVHcloud, 2026. https://www.ovhcloud.com/en/bare-metal/

10. Amazon Web Services. "Amazon SageMaker Pricing." AWS, 2026. https://aws.amazon.com/sagemaker/pricing/

11. Amazon Web Services. "EC2 Data Transfer Pricing." AWS, 2026. https://aws.amazon.com/ec2/pricing/on-demand/

12. Google Cloud. "Vertex AI Documentation." Google Cloud Docs, 2026. https://cloud.google.com/vertex-ai/docs

13. Microsoft. "Azure Machine Learning Documentation." Microsoft Learn, 2026. https://learn.microsoft.com/en-us/azure/machine-learning/

---

## Appendix: Note on Provider Selection

This analysis is based on a real project. Before this assignment, I evaluated AWS, Azure, GCP, OVHcloud, and Hetzner for a production system I am building. DigitalOcean was selected based on that evaluation.

The platform decomposition framework from lectures applies equally well to alternative providers, and a budget constrained perspective may offer a useful counterpoint to enterprise focused analyses.

I can discuss the architectural approaches and pricing models of AWS, Azure, and GCP based on my evaluation process.

---

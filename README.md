```
AUTHOR: Supriya Jadhav

Version: 1.0

Date: Dec 2025

© 2025 Supriya Sampat Jadhav. All rights reserved.
```

# Edge-IoT-AWS-Powered-Intelligent-Traffic-Light-Control


Built an edge-based IoT traffic light control solution integrated with AWS cloud analytics, enabling real-time vehicle detection, dynamic signal adjustments, and data-driven traffic optimization.


### Step by step guide to set up the Edge Lab Model 2 Demo

1)	Deploy CloudFormation template to provision resources and their dependencies in AWS cloud
2)	Run Python script on a G2 instance type using Deep Learning framework (PyTorch) for object detection in videos/images. 

#### Part 1:

- AWS CloudFormation template is used to deploy the resources to setup the edge demo lab model 2. 
- AWS CloudFormation is infrastructure as code service that provides an easy way to provision collection of related AWS resources and their dependencies in the cloud. To start with, describe a CloudFormation template with desired resources and their dependencies. Launch and configure them together as a stack. 


1. Describe/Pick a template (JSON or YAML) and define the requirements in your script

We used “EdgeLab - Single VPC, 1 EC2, 1 WL EC2 - ML.yaml” template that deployed following resources in AWS cloud environment:

    a)	VPC in N. Virginia Region
    b)	One Availability Zone: “us-east-1a” and the resources deployed in them are as follows: 
        i)	Public Subnet
        ii)	One - EC2 instance, instance type: t2.micro
        iii)AWS EC2 Security Group Ingress
        Add ingress rule to the security group of EC2 instance 
        iv)	AWS EC2 Security Group
        v)	Az Public Route Table
        vi)	Public Route
        vii)Subnet route table association

    c)	One Wavelength Zone: “us-east-1-wl1-atl-wlz-1” and the resources deployed in them are as follows:
        i)	Public Subnet
        ii)	One-DL EC2 instance, using AWS Deep Learning AMI, instance type: g4dn.2xlarge
        iii)	AWS EC2 Security Group Ingress
        Add ingress rule to the security group of the DL-EC2 instance
        iv)	AWS EC2 Security Gateway
        v)	Az Public Route Table
        vi)	Public Route
        vii)	Subnet route table association

    d)	Carrier Gateway
    e)	Internet Gateway
    f)	Gateway VPC Gateway Attachment

Refer to the Figure.1 below that describes AWS architecture diagram with all the resources and configurations explained above:

 
Figure.1: ATC Edge Demo Model 2






2.	Launch and configure the stack in AWS CloudFormation

    a)	Go to AWS CloudFormation console
    b)	Create the stack
    c)	Specify Template: Upload the template that we created in step.1 i.e. “EdgeLab - Single VPC, 1 EC2, 1 WL EC2 - ML.yaml”
    d)	Specify Stack details: Specify stack details like stack name, AWS Wavelength zone (AWS Best practice is to select a zone that is closer to the client’s location), EC2 key etc. Below are the specific configuration details used to set up the Edge Demo Lab:

    Stack Name: atc-edge-demo
    Current Wavelength Availability Zone: us-east-1-wl1-atl-wlz-1
    KeyName: yourkeyname.pem

    e)	Configure Stack Options: Configure stack options like Tags etc. 
    f)	Review and click create Stack. Wait until the status of the stack changes from in progress to create complete. After the stack is successfully created, all the resources will be up and running in the environment.

    NOTE: 
    •	It is recommended to use install cfn-lint which help to identify various errors in your stack before deploying it.   
    •	AWS CLI command validate-template only checks the syntax of the template.
    •	To read more about cfn-lint, click here

    g)	Install cfn-lint as follows. To read more about cfn-lint and it’s installation, click here

```
# Install Python 2.7+ and 3.4+ are supported.

pip install cfn-lint. If pip is not available, run python setup.py clean --all then python setup.py install.

# Homebrew (macOS)
brew install cfn-lint
```


Networking and Security configurations:

1.	Create an IAM role to enable the EC2 and S3 bucket communication. The EC2 instance needs to access various files stored in the AWS S3 bucket and hence we need to give the EC2 instance full AWS S3 access.

a.	Go to the IAM console and click on create role.
b.	Under AWS services, click EC2 and click next
c.	In attach permission policies page, search and select “AmazonS3FullAccess” and click next.
d.	Add tags, this part is optional. Click next
In review page, fill the details as below and click on create role.
Role Name: ec2-s3-full-access

2.	Attach an IAM role to the deep learning EC2 (instance type: g4dn.2xlarge) to be able to communicate with the S3 bucket

a.	Go to the EC2 console and select the deep learning EC2 (instance type: g4dn.2xlarge)
b.	Select actions -> Instance Settings->Attach/Replace IAM role. In the IAM role drop down menu, search and select ec2-s3-full-access and click apply.

3.	[NOTE: This step is required if you are using single CloudFormation template to spin up the resources. Skip this part if you are using nest stack cloud formation template].
Add a new inbound rule in the security group of deep learning instance (instance type: g4dn.2xlarge) so that it can communicate with the internet. Adding this rule will allow the installation of various libraries on the instance and run the object detection script successfully.

a.	In the EC2 console, go to the Security Group under Network and Security.
b.	Select the deep learning EC2 (instance type: g4dn.2xlarge) and go to the inbound rule section.
c.	Click edit and add a new inbound rule as follows:
Type: All Traffic
Protocol: All
Port Range: All
Source: 0.0.0.0/0

 


4.	Create Network interface and attach to the subnet deep learning EC2 (instance type: g4dn.2xlarge). The deep learning EC2 in public subnet does not have public IP address. 

a.	In EC2 console, select Network Interface under Network and Security. 
b.	Click on Create Network Interface, provide following details and click create.
Description: Its optional
Subnet: Select the subnet of deep learning EC2 (instance type: g4dn.2xlarge)

5.	Create and associate carrier IP to the deep learning EC2 (instance type: g4dn.2xlarge) running in wavelength zone.

a.	In EC2 console, select Elastic IPs under Network and Security.
b.	Click on Allocate new address, select the wavelength zone as per your requirement, leave the rest of the information as it is and click on allocate.
Network Border Group: us-east-1-wl1-atl-wlz-1
c.	In the same console, select Actions. It will display a few drop-down options, select Associate Address. Fill the below details and click Associate:
Resource Type: Network Interface
Network Interface: Network Interface ID that you created in Step 4
Private IP: Select respective private IP from drop-down list
 
Part 2:

1)	Copy the pem key from your local machine to the EC2 (instance type: t2.micro)

# Change key permission
chmod 400 supriya1.pem

# Upload the key to the EC2 
scp -i "supriya1.pem" "supriya1.pem" ec2-user@34.219.118.59:~/


2)	Connect to the EC2 (instance type: t2.micro) and confirm the pem key exists. Go to the AWS EC2 console, connect the EC2 (instance type: t2.micro) and run the following command. It should show the pem key

# list files in current directory
ls


3)	Connect to the deep learning EC2 (instance type: g4dn.2xlarge) from the EC2 (instance type: t2.micro)


ssh -i "supriya1.pem" ubuntu@10.192.1.180


4)	Set up aws configuration to be able to use AWS CLI


aws configure
AWS Access key ID [None]: You access key
AWS Secret Access Key [None]: Your secret access key
Default region name [none]: us-east-1
Default output format [None]: json





5)	Copy files from S3 bucket to the deep learning EC2 (instance type: t2.micro)

## Copying files from S3 bucket to the EC2
aws s3 cp s3://atc-edge-demo/Predictor.py Predictor.py
aws s3 cp s3://atc-edge-demo/test.jpg test.jpg



6)	Install the Detectron2 library. To read more about various ways of installation, click here

# Select the below environmnet
Source activate pytorch_latest_p37



Python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2



7)	Run python script to perform object detection in videos/images.


# Input give is an Image file
python Predictor.py --input test.jpg

# Input given is a Video file
python Predictor.py --video-input test_video.mp4





References:
1.	https://aws.amazon.com/cloudformation/
2.	https://aws.amazon.com/premiumsupport/knowledge-center/ec2-wavelength-zone-connection-errors/
3.	https://aws.amazon.com/blogs/mt/git-pre-commit-validation-of-aws-cloudformation-templates-with-cfn-lint/
4.	https://formulae.brew.sh/formula/cfn-lint
5.	https://dev.to/namuny/using-cfn-lint-to-validate-your-cloudformation-template-jpa
6.	https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html
7.	https://detectron2.readthedocs.io/en/latest/tutorials/install.html


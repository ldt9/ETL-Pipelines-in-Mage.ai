# PEDT PIPE Repository

> [!CAUTION]
> Do not add or modify files in the `main` branch under any circumstance.

## Development Instructions
1. Create a new branch following the proper naming conventions (outlined below).
2. Make changes only in your new branch
3. Create a "Pull Request" when you want your changes reviewed.
4. Wait for changes to be reviewed and approved/rejected.

### Branch name guidelines:
- bugfix/your-branch
- dev/your-branch
- release/your-branch

### Setup Instructions:
1. Must have [Docker](https://docs.docker.com/get-docker/) installed prior to running the setup commands.
2. Run `docker run -it -p 6789:6789 mageprod:latest /app/run_app.sh mage start etl` and/or use this [link](https://docs.mage.ai/production/ci-cd/local-cloud/repository-setup) to setup your environment.

Read more aboute [mage.ai](https://github.com/mage-ai/mage-ai/tree/master) here.

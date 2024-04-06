# Overview
This repository is used to manage ETL pipelines that automate updating a MySQL database. Transactions can be housed as a `.csv` in the `etl\data` folder. Portfolio return calculations are based off the logic found in this [repository](https://github.com/ldt9/Portfolio-Tearsheet-Return-Generator/blob/main/Portfolio_Tear_Sheet_Generator_with_Sector_Performace.ipynb)

### Setup Instructions:
1. Must have [Docker](https://docs.docker.com/get-docker/) installed prior to running the setup commands.
2. Run `docker run -it -p 6789:6789 mageprod:latest /app/run_app.sh mage start etl` and/or use this [link](https://docs.mage.ai/production/ci-cd/local-cloud/repository-setup) to setup your environment.

Read more aboute [mage.ai](https://github.com/mage-ai/mage-ai/tree/master) here.

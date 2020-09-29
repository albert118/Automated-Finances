# Automated-Finances

This project encompasses development of several finance automation, visualisation and management tools. Designed for personal (single user) use. Currently this supports:
* scraping NetBank data and preprocessing (then importing) data from NetBank as well as [ADP](https://developers.adp.com/metadata/collections/all/learn)(payroll service),
* calculating several useful statistics from the data,
* configurable categories and subcategories for expenditure, income and savings data,
* category and subcategory visualisations for expenditure, income and savings data (always more coming!),
* some dynamic control over the dataframe management to reduce hard coding labels, headers, etc... (more to come).

## Current Features:
### Monthly overview of bank accounts
* incoming review; avg., total p/mnth, total p each week, by category:
* outgoing overview; total, avg., by category 
(coffee, shopping, eating, groceries, utilities, health)
* savings overview: avg., total p/mnth, total p each week

### Paycheck integrations from ADP payroll solutions
* last received,
* hourly and commission based stats

# Planned Update

Given my recent foray into Javascript, React and making it work with Django, I decided on a revamp to bring everything together. Since this project was always a report generating tool I planned to run permanently and have sent to me, having it run as webservice on a server makes sense.

These are the mock ups I've developed so far, and what is planned for the next update.

![Income View page #1](designSpecs/viewMockUps/Incomes%20(view)%20mock%20up/1.jpg)

![Income View page #2](designSpecs/viewMockUps/Incomes%20(view)%20mock%20up/2.jpg)

![Income View page #3](designSpecs/viewMockUps/Incomes%20(view)%20mock%20up/4.jpg)

![Budget View page #1](designSpecs/viewMockUps/Budgets%20(view)%20mock%20up/1.jpg)

![Budget View page #2](designSpecs/viewMockUps/Budgets%20(view)%20mock%20up/2.jpg)

![Budget View page #3](designSpecs/viewMockUps/Budgets%20(view)%20mock%20up/3.jpg)


## Planned features will include:
### Monthly overview of bank accounts
* opening balance : closing balance : delta
* Category and subcategory visualisations to help track progress to 
 goals and see current position!
* planned/upcoming expenditure integrations (see calendar integrations)
* user savings goals and calculator tools for interest - fee accumulations

### Paycheck integrations from ADP payroll solutions
* 4-week average,
    
### Events and Calendar Integrations
* facebook event tracking (group specific)
* iCal/Google Cal integrations
* other??

### Superannuaton Interation
* Review of fees
* Investment gains/losses
* Overview of investment strategy(ies)


# TODOs:

* TODO: Fix for ubuntu emulator with 'mnt' path logic
* TODO: Find an alt. for tabula that doesn't require Java

# Notes:

* Java 8+ required for tabula.

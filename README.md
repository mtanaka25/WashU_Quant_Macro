# WashU_Quant_Macro
These codes are for the assignments of Economics 5725 (Quantitative Macroeconomic Theory, Fall 2022) at the Washignton University in St. Louis.


## About Economics 5725

Economics 5725 is a PhD lecture taught by Juan M. Sanchez (Federal Reserve Bank of St. Louis).

The goals of this course are (1) to introduce the basics tools of modern quantitative macroeconomics, and (2) to help us to start working on quantitative macro.
In the course, we are expected to implement models with rich
heterogeneity across agents and firms, aggregate fluctuations, and transitional dynamics.
For the detail of the course, see the syllabus available on [Juan's personal website](https://sites.google.com/view/juanmsanchezweb).

Because the PDFs of the homework assignments are copyrighted by the instructor, I will not share them. The only things I will upload here are the codes I write for my own solutions.


## Programming language
 I basically use Python, which is recommended by the lecturer. While I have not done so for now, however, I might occasionally use MATLAB when I think it is more suited. 


## Requirements
I use the latest version of Python and its basic packages (e.g. numpy). I write the scripts with the latest version of Visual Studio Code and run them on the latest Mac OS's Terminal.
Note that, in case I would write MATLAB codes, I would use R2022b.

In the codes in `HW1` and `HW2`, I use `freadapi` and `statsmodels` packages, which are not included in standard collections of Python packages. To use `freadapi`, you need to apply a API key to St. Louis Fed. The application is accepted online for free. 

After finishing `HW5`, I cut out several routines which are often used in the assignments as an independent package. Thus, for `HW6` and later, you need `mtPythonTools`, which is available in [the corresponding repository](https://github.com/mtanaka25/mtPythonTools).  

## Cautions
In `HW4`, we were asked to implement the policy iteration based on the Euler equation and the endogenous grid point method (see [Carroll [2006, *Econ. Lett.*]](https://doi.org/10.1016/j.econlet.2005.09.013) for the EGP method). Although I finished writing scripts that looked to work well, however, I failed to get reasonable solutions. So, please doublecheck if these codes are correct when you reuse them.

In `HW7`, we were asked to solve a variant of [Athreya, Mustre-del-Rio, and Sanchez [2019, *Rev. Financ. Stud.*]](https://academic.oup.com/rfs/article/32/10/3851/5305595) model. However, due to the time constraint, I could not affort to reach the fully reasonable ansewrs. In particular, the statistics for the whole working-age population are not consistent with age-by-age statistics. I would fix it if I have time to do so. In the meantime, please carefully doublecheck the codes if you reuse them.

## Disclaimer
This is my own programs used for my homework. Thus, it is not guaranteed that these codes are correct and what the lecturer expected. 
My priority is to meet the submission deadlines. So, some functions in the programs might lack generality. Please be careful about that when you recycle any scripts in this repository.

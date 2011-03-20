
/*
 *
 *    MultiBoost - Multi-purpose boosting package
 *
 *    Copyright (C) 2010   AppStat group
 *                         Laboratoire de l'Accelerateur Lineaire
 *                         Universite Paris-Sud, 11, CNRS
 *
 *    This file is part of the MultiBoost library
 *
 *    This library is free software; you can redistribute it 
 *    and/or modify it under the terms of the GNU General Public
 *    License as published by the Free Software Foundation; either
 *    version 2.1 of the License, or (at your option) any later version.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin St, 5th Floor, Boston, MA 02110-1301 USA
 *
 *    Contact: Balazs Kegl (balazs.kegl@gmail.com)
 *             Norman Casagrande (nova77@gmail.com)
 *             Robert Busa-Fekete (busarobi@gmail.com)
 *
 *    For more information and up-to-date version, please visit
 *        
 *                       http://www.multiboost.org/
 *
 */


/**
 * \file VJCascadeLearner.h The meta-learner AdaBoostLearner.MH.
 */
#pragma warning( disable : 4786 )

#ifndef __VJ_CASCADE_LEARNER_H
#define __VJ_CASCADE_LEARNER_H

#include "StrongLearners/GenericStrongLearner.h"
#include "Utils/Args.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	
	class OutputInfo;
	class BaseLearner;
	class InputData;
	class Serialization;
	
	/**
	 * The AdaBoost learner. This class performs the meta-learning
	 * by calling the weak learners and updating the weights.
	 * \date 12/11/2005
	 */
	class VJCascadeLearner : public GenericStrongLearner
	{
	public:
		
		/**
		 * The constructor. It initializes the variables and sets them using the
		 * information provided by the arguments passed. They are parsed
		 * using the helpers provided by class Args.
		 * \date 13/11/2005
		 */
		VJCascadeLearner()
		: _numIterations(0), _maxTime(-1), _verbose(1), _smallVal(1E-10), _stageStartNumber(2),
        _resumeShypFileName(""), _outputInfoFile(""), _withConstantLearner(false), _maxAcceptableFalsePositiveRate(0.05), _minAcceptableDetectionRate(0.95){}
		
		/**
		 * Start the learning process.
		 * \param args The arguments provided by the command line with all
		 * the options for training.
		 * \see OutputInfo
		 * \date 10/11/2005
		 */
		virtual void run(const nor_utils::Args& args);
		
		/**
		 * Performs the classification using the AdaBoostMHClassifier.
		 * \param args The arguments provided by the command line with all
		 * the options for classification.
		 */
		virtual void classify(const nor_utils::Args& args);
		
		/**
		 * Print to stdout (or to file) a confusion matrix.
		 * \param args The arguments provided by the command line.
		 * \date 20/3/2006
		 */
		virtual void doConfusionMatrix(const nor_utils::Args& args);
		
		/**
		 * Output the outcome of the strong learner for each class.
		 * Strictly speaking these are (currently) not posteriors,
		 * as the sum of these values is not one.
		 * \param args The arguments provided by the command line.
		 */
		virtual void doPosteriors(const nor_utils::Args& args);
		
		
		/**
		 * Output the AUC values for each class. In multi-class scenario, 
		 * one particualar class is the positive one and the rest ones are considered
		 * as negative one.
		 * \param args The arguments provided by the command line.
		 */
		virtual void doROC(const nor_utils::Args& args);
		
		/**
		 * Output the class conditional probilities of the strong learner for each class.
		 * the calibration is based on Platt's method.
		 * 
		 * \param args The arguments provided by the command line.
		 */
		virtual void doCalibratedPosteriors(const nor_utils::Args& args);
		
		
		/**
		 * Output the likelihood of the strong learner for each iteration.
		 * \param args The arguments provided by the command line.
		 */
		virtual void doLikelihoods(const nor_utils::Args& args);
						
		
		virtual float updateWeights(InputData* pData, BaseLearner* pWeakHypothesis);
		
		virtual void resetWeights(InputData* pData);
	protected:
		
		/**
		 * Get the needed parameters (for the strong learner) from the argumens.
		 * \param The arguments provided by the command line.
		 */
		void getArgs(const nor_utils::Args& args);
		
		
		/**
		 * Resume the weak learner list.
		 * \return The current iteration number. 0 if not -resume option has been called
		 * \param pTrainingData The pointer to the training data, needed for classMap, enumMaps.
		 * \date 21/12/2005
		 * \see resumeProcess
		 * \remark resumeProcess must be called too!
		 */
		int resumeWeakLearners(InputData* pTrainingData);
		
		
		void calculatePosteriors( InputData* pData, vector<BaseLearner*>& weakhyps, vector<double>& posteriors );
		
		vector<vector<BaseLearner*> >  _foundHypotheses; //!< The list of the hypotheses found.
		vector<vector<double> >  _threshold; //!< The list of the hypotheses found.
		
		string  _baseLearnerName; //!< The name of the basic learner used by AdaBoost. 
		string  _shypFileName; //!< File name of the strong hypothesis.
		bool	   _isShypCompressed; 
		
		string  _trainFileName;
		string  _validFileName;
		string  _testFileName;
		
		int     _numIterations;
		int     _maxTime; //!< Time limit for the whole processing. Default: no time limit (-1).		
		int		_stageStartNumber;
		/**
		 * Verbose level.
		 * There are three levels of verbosity:
		 * - 0 = no messages
		 * - 1 = basic messages
		 * - 2 = show all messages
		 */
		int     _verbose;
		const float _smallVal; //!< A small value, to solve numeric issues
		
		/**
		 * If resume is set, this will hold the strong hypothesis file to load in order to 
		 * continue with the training process.
		 */
		string  _resumeShypFileName;
		string  _outputInfoFile; //!< The filename of the step-by-step information file that will be updated

		
		bool	_withConstantLearner;

		double _maxAcceptableFalsePositiveRate; // f
		double _minAcceptableDetectionRate; // d
		
		
		////////////////////////////////////////////////////////////////
	private:
		/**
		 * Fake assignment operator to avoid warning.
		 * \date 6/12/2005
		 */
		VJCascadeLearner& operator=( const VJCascadeLearner& ) {return *this;}
		
		/**
		 * A temporary variable for h(x)*y. Helps saving time during re-weighting.
		 */				
		vector< vector<float> > _hy;
	};
	
} // end of namespace MultiBoost

#endif // __ADABOOST_MH_LEARNER_H

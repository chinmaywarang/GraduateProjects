/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

//==============================================================================
/**
*/
class DistortionPedalAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                             public juce::Slider::Listener

{
public:
    DistortionPedalAudioProcessorEditor (DistortionPedalAudioProcessor&);
    ~DistortionPedalAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    void sliderValueChanged(juce::Slider* slider) override;
    
private:
    
    DistortionPedalAudioProcessor& audioProcessor;
    juce::Slider mGainSlider;
    juce::Slider mVolumeSlider;
    juce::ImageComponent mImageComponent;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DistortionPedalAudioProcessorEditor)
};

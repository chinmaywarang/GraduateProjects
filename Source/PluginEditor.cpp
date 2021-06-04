/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
DistortionPedalAudioProcessorEditor::DistortionPedalAudioProcessorEditor (DistortionPedalAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize (200, 200);
    auto image = juce::ImageCache::getFromMemory(BinaryData::maxresdefault_jpg, BinaryData::maxresdefault_jpgSize);
    if (! image.isNull())
    {
      mImageComponent.setImage( image,juce::RectanglePlacement::stretchToFit);
    }
    else
    {
        jassert(! image.isNull());
    }
    addAndMakeVisible(mImageComponent);
    mGainSlider.setSliderStyle (juce::Slider::SliderStyle::RotaryHorizontalVerticalDrag);
    mGainSlider.setTextBoxStyle(juce::Slider::TextBoxRight, true, 30, 20);
    mGainSlider.setRange(0.0f, 1.0f,0.01f);
    mGainSlider.setValue(0.0f);
    mGainSlider.addListener(this);
    addAndMakeVisible(mGainSlider);
    mVolumeSlider.setSliderStyle (juce::Slider::SliderStyle::RotaryHorizontalVerticalDrag);
    mVolumeSlider.setTextBoxStyle(juce::Slider::TextBoxRight, true, 30, 20);
    mVolumeSlider.setRange(0.0f, 1.0f,0.01f);
    mVolumeSlider.setValue(1.0f);
    mVolumeSlider.addListener(this);
    addAndMakeVisible(mVolumeSlider);
    getLookAndFeel().setColour(juce::Slider::thumbColourId, juce::Colours::purple);
    getLookAndFeel().setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colours::darkgrey);
    getLookAndFeel().setColour(juce::Slider::rotarySliderFillColourId, juce::Colours::yellow);

    
    
    

    
}

DistortionPedalAudioProcessorEditor::~DistortionPedalAudioProcessorEditor()
{
}

//==============================================================================
void DistortionPedalAudioProcessorEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (juce::Colours::black);
    
    
}

void DistortionPedalAudioProcessorEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
    mGainSlider.setBounds(getWidth()/2 - 50,getHeight()/2 - 90,110,110);
    mVolumeSlider.setBounds(getWidth()/2 - 50,getHeight() - 90,110,110);
    mImageComponent.setBoundsRelative(0.0f, 0.0f, 0.20f, 0.40f);
}
void DistortionPedalAudioProcessorEditor::sliderValueChanged(juce::Slider *slider)
{
    if (slider == &mGainSlider)
    {
        audioProcessor.mGain = mGainSlider.getValue();
    }
    if (slider == &mVolumeSlider)
    {
        audioProcessor.mVolume = mVolumeSlider.getValue();
    }

}
